"""Generate candidate entrance points for each building.

For each building, combines positive entrance samples (real entrances from OSM)
with negative samples (randomly generated fake candidates on the building perimeter).
Ensures each building has exactly n_candidates total candidate entrances.

The candidate points are placed by:
1. Using existing entrance points from entrance_geometries as positive samples
2. Generating negative samples by:
   - Sampling points along the building's perimeter (exterior ring)
   - Points are distributed uniformly at random positions along the boundary
   - If insufficient points after negatives, uses points slightly inside/outside the boundary

Expected config keys in config.yaml:
- processed_data: path to processed data root
- cities: list of city names

Output: Updates metadata.gpkg with a new "candidates" column containing:
    {
        "positive": [Point, Point, ...],  (real entrances)
        "negative": [Point, Point, ...],  (generated negatives)
        "labels": [1, 1, ..., 0, 0, ...]  (1 for positive, 0 for negative)
    }
"""

import ast
import hashlib
import json
import os
import random
import re

import geopandas as gpd
import numpy as np
import yaml
from shapely import wkt
from shapely.geometry import Point
from tqdm import tqdm


def sample_points_on_boundary(geometry, n_points, random_seed=None):
    """Sample n_points uniformly at random positions along the geometry's boundary.

    Args:
        geometry: Shapely geometry (Polygon or MultiPolygon)
        n_points: Number of points to sample
        random_seed: Random seed for reproducibility

    Returns:
        List of Point geometries
    """
    if random_seed is not None:
        random.seed(random_seed)

    boundary = geometry.boundary

    # Handle both LinearRing (from Polygon) and MultiLineString (from MultiPolygon)
    if hasattr(boundary, "length"):
        # Single boundary
        boundaries = [boundary]
    else:
        # Multiple boundaries (from MultiPolygon)
        boundaries = list(boundary.geoms) if hasattr(boundary, "geoms") else [boundary]

    # Calculate cumulative lengths
    cumulative_lengths = []
    total_length = 0
    for b in boundaries:
        total_length += b.length
        cumulative_lengths.append(total_length)

    if total_length == 0:
        # Degenerate geometry, return centroid
        return [geometry.centroid] * n_points

    points = []
    for _ in range(n_points):
        # Sample a random distance along the total boundary
        sampled_distance = random.uniform(0, total_length)

        # Find which boundary segment contains this distance
        for i, cum_length in enumerate(cumulative_lengths):
            if sampled_distance <= cum_length:
                boundary_seg = boundaries[i]
                # Distance within this segment
                if i == 0:
                    dist_in_seg = sampled_distance
                else:
                    dist_in_seg = sampled_distance - cumulative_lengths[i - 1]
                break

        # Get the point at this distance
        point = boundary_seg.interpolate(dist_in_seg)
        points.append(point)

    return points


def generate_negative_samples(building_geom, n_negatives, random_seed=None):
    """Generate n_negatives fake entrance candidates.

    Args:
        building_geom: Shapely Polygon or MultiPolygon
        n_negatives: Number of negative samples to generate
        random_seed: Random seed for reproducibility

    Returns:
        List of Point geometries
    """
    if n_negatives <= 0:
        return []

    # Sample points on the boundary
    negative_points = sample_points_on_boundary(
        building_geom, n_negatives, random_seed=random_seed
    )

    return negative_points


def parse_entrance_geometries(value):
    """Parse serialized entrance geometries into a list of Points.

    Step 1 stores entrance geometries in metadata as a stringified Python list
    of WKT strings for GeoPackage compatibility.
    """
    if value is None:
        return []

    # Already parsed as an in-memory iterable of geometry-like objects.
    if isinstance(value, (list, tuple)):
        parsed = []
        for item in value:
            if isinstance(item, Point):
                parsed.append(item)
            elif isinstance(item, str):
                try:
                    geom = wkt.loads(item)
                    if isinstance(geom, Point):
                        parsed.append(geom)
                except Exception:
                    continue
        return parsed

    if not isinstance(value, str):
        return []

    text = value.strip()
    if not text:
        return []

    # Try Python-literal form first (e.g. "['POINT (...)', 'POINT (...)']").
    items = None
    try:
        items = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        # Try JSON form (e.g. '["POINT (...)", "POINT (...)"]').
        try:
            items = json.loads(text)
        except (TypeError, ValueError):
            items = None

    # Last-resort: extract WKT POINT tokens directly from free-form strings.
    if items is None:
        wkts = re.findall(r"POINT\s*\([^\)]*\)", text, flags=re.IGNORECASE)
        items = wkts

    if not isinstance(items, list):
        return []

    parsed = []
    for item in items:
        if not isinstance(item, str):
            continue
        try:
            geom = wkt.loads(item)
            if isinstance(geom, Point):
                parsed.append(geom)
        except Exception:
            continue

    return parsed


def serialize_candidates(positive_samples, negative_samples):
    """Serialize candidate points into a GeoPackage-safe JSON string."""
    all_candidates = positive_samples + negative_samples
    labels = [1] * len(positive_samples) + [0] * len(negative_samples)

    payload = {
        "positive": [pt.wkt for pt in positive_samples],
        "negative": [pt.wkt for pt in negative_samples],
        "labels": labels,
        "all_points": [pt.wkt for pt in all_candidates],
    }
    return json.dumps(payload)


def add_candidates_one_city(city, n_candidates):
    """Process one city: generate candidate entrance points."""
    print(f"Processing city: {city}")
    metadata_path = os.path.join(processed_path, city, "metadata.gpkg")

    if not os.path.exists(metadata_path):
        print(f"  Warning: metadata not found at {metadata_path}")
        return

    gdf = gpd.read_file(metadata_path, layer="Filtered_Buildings")

    if "entrance_geometries" not in gdf.columns:
        raise ValueError(
            "Required column 'entrance_geometries' not found in metadata layer "
            f"for {city}. Available columns: {gdf.columns.tolist()}"
        )

    candidates_list = []
    n_insufficient = 0

    for idx, row in tqdm(
        enumerate(gdf.itertuples()), total=len(gdf), desc=f"  Generating {city}"
    ):
        # Get positive samples (real entrances)
        if hasattr(row, "entrance_geometries") and row.entrance_geometries:
            positive_samples = parse_entrance_geometries(row.entrance_geometries)
        else:
            positive_samples = []

        n_positive = len(positive_samples)
        n_negative = n_candidates - n_positive

        # Generate negative samples
        if n_negative > 0:
            # Use building index as part of random seed for reproducibility.
            # MD5 is used here purely for deterministic hashing (not cryptography);
            # this ensures consistent candidate generation regardless of PYTHONHASHSEED.
            seed = int.from_bytes(
                hashlib.md5(f"{city}:{idx}".encode()).digest()[:4], "little"
            )
            negative_samples = generate_negative_samples(
                row.geometry, n_negative, random_seed=seed
            )
        else:
            negative_samples = []
            if n_positive > n_candidates:
                n_insufficient += 1
                # Truncate excess positive samples
                positive_samples = positive_samples[:n_candidates]
                n_positive = len(positive_samples)

        candidates_list.append(serialize_candidates(positive_samples, negative_samples))

    gdf["candidates"] = candidates_list

    # Print statistics
    total_positive = 0
    total_negative = 0
    for payload in candidates_list:
        decoded = json.loads(payload)
        total_positive += len(decoded.get("positive", []))
        total_negative += len(decoded.get("negative", []))

    print(f"  Total buildings: {len(gdf)}")
    print(f"  Total positive samples: {total_positive}")
    print(f"  Total negative samples: {total_negative}")
    print(
        f"  Average samples per building: {(total_positive + total_negative) / len(gdf):.2f}"
    )

    if n_insufficient > 0:
        print(
            f"  Warning: {n_insufficient} buildings had more positive samples than n_candidates"
        )

    gdf.to_file(metadata_path, layer="Filtered_Buildings", driver="GPKG")
    print(f"  Updated metadata: {metadata_path}\n")


def add_candidates(cities, n_candidates):
    """Add candidate entrance points for all cities."""
    for city in cities:
        add_candidates_one_city(city, n_candidates=n_candidates)


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    processed_path = config["processed_data"]
    cities = config["cities"]
    n_candidates = config["n_candidates"]

    add_candidates(cities, n_candidates=n_candidates)
