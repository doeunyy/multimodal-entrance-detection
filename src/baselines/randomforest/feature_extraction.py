"""
Feature extraction for each candidate sample on a building footprint.

With the available data (no separate road/context layers), we extract:
  - Intrinsic features from the footprint geometry
  - Inter-building features from nearby buildings in the same dataset
  - Address-derived features

Features marked (*) also get a within-building normalised rank feature.
"""
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from shapely.affinity import scale
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.ops import unary_union


# ===================================================================
# Geometry helpers
# ===================================================================

def _outer_perpendicular_line(
    sample: Point, master_edge: LineString, footprint: Polygon, length: float = 300.0,
) -> LineString:
    """Line from sample perpendicular to master edge, pointing outward."""
    dx = master_edge.coords[-1][0] - master_edge.coords[0][0]
    dy = master_edge.coords[-1][1] - master_edge.coords[0][1]
    elen = math.hypot(dx, dy)
    if elen < 1e-9:
        return LineString([sample, sample])
    nx, ny = -dy / elen, dx / elen  # one normal direction

    # Test which side is outward
    test = Point(sample.x + nx * 0.5, sample.y + ny * 0.5)
    if footprint.contains(test):
        nx, ny = -nx, -ny  # flip to outward
    end = Point(sample.x + nx * length, sample.y + ny * length)
    return LineString([sample, end])


def _angle_at_vertex(coords, idx) -> float:
    """Interior angle (degrees) at vertex idx in a closed ring."""
    n = len(coords) - 1
    p0 = np.array(coords[(idx - 1) % n])
    p1 = np.array(coords[idx % n])
    p2 = np.array(coords[(idx + 1) % n])
    v1, v2 = p0 - p1, p2 - p1
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def _edge_concavity(ring_coords, edge_idx) -> float:
    """Classify edge: 1.0=convex, 0.0=concave, -1.0=neither."""
    n = len(ring_coords) - 1
    if n < 4:
        return -1.0
    i1 = edge_idx % n
    i2 = (edge_idx + 1) % n
    a1 = _angle_at_vertex(ring_coords, i1)
    a2 = _angle_at_vertex(ring_coords, i2)
    tol = 30
    if abs(a1 - 90) < tol and abs(a2 - 90) < tol:
        return 1.0
    if abs(a1 - 270) < tol and abs(a2 - 270) < tol:
        return 0.0
    return -1.0


def _reflection_axis(footprint: Polygon) -> Optional[LineString]:
    """
    Heuristic symmetry axis via the oriented bounding box long-axis.
    Returns the axis LineString if the footprint is roughly symmetric, else None.
    """
    obb = footprint.minimum_rotated_rectangle
    c = list(obb.exterior.coords)
    e1 = LineString([c[0], c[1]])
    e2 = LineString([c[1], c[2]])
    if e1.length >= e2.length:
        m1 = e1.interpolate(0.5, normalized=True)
        m2 = LineString([c[2], c[3]]).interpolate(0.5, normalized=True)
    else:
        m1 = e2.interpolate(0.5, normalized=True)
        m2 = LineString([c[3], c[0]]).interpolate(0.5, normalized=True)
    axis = LineString([m1, m2])
    origin = axis.interpolate(0.5, normalized=True)
    # Check approximate symmetry via IoU of footprint and its reflection
    try:
        reflected = scale(footprint, xfact=-1, origin=origin)
        iou = footprint.intersection(reflected).area / footprint.union(reflected).area
        return axis if iou > 0.85 else None
    except Exception:
        return None


def _opposite_edge_shape(footprint: Polygon, master_edge: LineString, ring_coords) -> float:
    """Concavity of the edge opposite to the master edge."""
    mid = master_edge.interpolate(0.5, normalized=True)
    dx = -(master_edge.coords[-1][1] - master_edge.coords[0][1])
    dy = master_edge.coords[-1][0] - master_edge.coords[0][0]
    norm = math.hypot(dx, dy) + 1e-12
    dx /= norm
    dy /= norm

    # Determine inward direction
    test = Point(mid.x + dx * 0.5, mid.y + dy * 0.5)
    if not footprint.contains(test):
        dx, dy = -dx, -dy
    ray = LineString([mid, Point(mid.x + dx * 500, mid.y + dy * 500)])
    ring_line = LineString(footprint.exterior.coords)
    inter = ray.intersection(ring_line)
    if inter.is_empty:
        return -1.0

    # Collect intersection points
    hit_pts = []
    if isinstance(inter, Point):
        hit_pts = [inter]
    elif hasattr(inter, "geoms"):
        hit_pts = [g for g in inter.geoms if isinstance(g, Point)]
    if not hit_pts:
        return -1.0

    farthest = max(hit_pts, key=lambda p: mid.distance(p))
    # Find which edge it lands on
    coords = list(footprint.exterior.coords)
    for eidx in range(len(coords) - 1):
        edge = LineString([coords[eidx], coords[eidx + 1]])
        if edge.distance(farthest) < 0.5:
            return _edge_concavity(ring_coords, eidx)
    return -1.0


# ===================================================================
# Main extraction
# ===================================================================

def extract_features_for_building(
    samples: List[Point],
    edge_indices: List[int],
    master_edges: List[LineString],
    footprint: Polygon,
    nearby_buildings: Optional[List[Polygon]],
    has_addr_street: bool,
) -> pd.DataFrame:
    """
    Extract features for all samples of one building.

    Parameters
    ----------
    samples : candidate entrance points along the perimeter
    edge_indices : which polygon edge each sample sits on
    master_edges : the polygon's exterior edges
    footprint : the building's footprint polygon (projected, metres)
    nearby_buildings : list of other building footprints within the buffer
    has_addr_street : whether the building has an addr:street tag

    Returns
    -------
    DataFrame with one row per sample and one column per feature.
    """
    n = len(samples)
    centroid = footprint.centroid
    ring_coords = list(footprint.exterior.coords)
    sym_axis = _reflection_axis(footprint)

    # Precompute obstacle union for open-area and visibility
    obstacle_union = None
    if nearby_buildings:
        valid = [b for b in nearby_buildings if b is not None and b.is_valid]
        if valid:
            obstacle_union = unary_union(valid)

    records = []
    for i in range(n):
        pt = samples[i]
        eidx = edge_indices[i]
        me = master_edges[eidx]
        f: Dict[str, Optional[float]] = {}

        # --- Intrinsic features ---

        # Distance to centroid (*)
        f["to_centroid"] = pt.distance(centroid)

        # Proportion: closeness of sample to midpoint of its master edge (*)
        me_len = me.length
        me_mid = me.interpolate(0.5, normalized=True)
        f["proportion"] = pt.distance(me_mid) / me_len if me_len > 0 else 0.0

        # Length of master edge (*)
        f["master_edge_len"] = me_len

        # Symmetry axis features
        f["has_axis"] = 1.0 if sym_axis is not None else 0.0
        f["dist_to_axis"] = pt.distance(sym_axis) if sym_axis else np.nan
        f["at_axis_edge"] = (
            1.0 if (sym_axis and me.intersects(sym_axis)) else 0.0
        ) if sym_axis else np.nan

        # Face inner: does the OPL cross back into the footprint?
        opl = _outer_perpendicular_line(pt, me, footprint, length=200.0)
        ring_line = LineString(footprint.exterior.coords)
        # Count intersections excluding the start point vicinity
        opl_inter = opl.intersection(ring_line)
        f["face_inner"] = 0.0
        if not opl_inter.is_empty:
            if isinstance(opl_inter, Point):
                if pt.distance(opl_inter) > 1.0:
                    f["face_inner"] = 1.0
            elif hasattr(opl_inter, "geoms"):
                far_pts = [g for g in opl_inter.geoms
                           if isinstance(g, Point) and pt.distance(g) > 1.0]
                f["face_inner"] = 1.0 if far_pts else 0.0

        # Concavity / convexity
        f["concavity"] = _edge_concavity(ring_coords, eidx)

        # Opposite edge shape
        f["oppo_shape"] = _opposite_edge_shape(footprint, me, ring_coords)

        # Normalised position along the full perimeter
        ring_ls = LineString(footprint.exterior.coords)
        f["norm_position"] = ring_ls.project(pt) / ring_ls.length

        # Edge angle relative to longest edge (captures orientation)
        longest_edge = max(master_edges, key=lambda e: e.length)
        le_dx = longest_edge.coords[-1][0] - longest_edge.coords[0][0]
        le_dy = longest_edge.coords[-1][1] - longest_edge.coords[0][1]
        me_dx = me.coords[-1][0] - me.coords[0][0]
        me_dy = me.coords[-1][1] - me.coords[0][1]
        dot = le_dx * me_dx + le_dy * me_dy
        cross = le_dx * me_dy - le_dy * me_dx
        f["edge_angle_to_longest"] = math.atan2(cross, dot)

        # --- Inter-building (extrinsic) features ---

        # Distance to nearest other building (*)
        if obstacle_union is not None:
            f["dist_to_nearest_bldg"] = pt.distance(obstacle_union)
        else:
            f["dist_to_nearest_bldg"] = np.nan

        # Open area: distance along OPL to the first obstacle (*)
        if obstacle_union is not None:
            opl_obs = opl.intersection(obstacle_union)
            if opl_obs.is_empty:
                f["open_area"] = opl.length
            elif isinstance(opl_obs, Point):
                f["open_area"] = pt.distance(opl_obs)
            elif hasattr(opl_obs, "geoms"):
                pts = [g for g in opl_obs.geoms if isinstance(g, Point)]
                f["open_area"] = min((pt.distance(p) for p in pts), default=opl.length)
            else:
                # LineString intersection etc.
                f["open_area"] = pt.distance(Point(opl_obs.coords[0]))
        else:
            f["open_area"] = np.nan

        # --- Address feature ---
        f["has_addr_street"] = 1.0 if has_addr_street else 0.0

        records.append(f)

    df = pd.DataFrame(records)

    # --- Add normalised within-building rank features for (*) columns ---
    rank_base_cols = [
        "to_centroid", "proportion", "master_edge_len",
        "dist_to_axis", "dist_to_nearest_bldg", "open_area",
    ]
    for col in rank_base_cols:
        if col not in df.columns:
            continue
        vals = df[col].values.astype(float)
        valid = ~np.isnan(vals)
        ranks = np.full(n, np.nan)
        if valid.sum() > 0:
            order = np.argsort(vals[valid])
            r = np.empty_like(order, dtype=float)
            r[order] = np.arange(1, valid.sum() + 1)
            ranks[valid] = r / valid.sum()
        df[f"{col}_rank"] = ranks

    return df