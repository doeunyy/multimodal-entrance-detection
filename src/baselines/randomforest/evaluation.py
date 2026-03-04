"""
Evaluation metrics for entrance tagging.

- Linear distance error: shortest distance along the perimeter
- Path distance error: Euclidean (no routing graph available)
- Absolute and relative ranking of true entrance among predictions
"""
from typing import Dict, List

import numpy as np
from shapely.geometry import LineString, Point, Polygon


def linear_distance_error(
    footprint: Polygon, true_pt: Point, estimated_pt: Point,
) -> float:
    """Shortest distance along the perimeter between two points."""
    ring = LineString(footprint.exterior.coords)
    d1 = ring.project(true_pt)
    d2 = ring.project(estimated_pt)
    along = abs(d1 - d2)
    return min(along, ring.length - along)


def euclidean_distance_error(true_pt: Point, estimated_pt: Point) -> float:
    return true_pt.distance(estimated_pt)


def ranking_metrics(proba: np.ndarray, true_pos_idx: int) -> Dict[str, float]:
    """Absolute and relative rank of the true entrance sample (1-based)."""
    n = len(proba)
    sorted_idx = np.argsort(-proba)
    abs_rank = int(np.where(sorted_idx == true_pos_idx)[0][0]) + 1
    return {"absolute_rank": abs_rank, "relative_rank": abs_rank / n}


def compute_building_metrics(
    footprint: Polygon,
    true_pt: Point,
    estimated_pt: Point,
    proba: np.ndarray,
    true_pos_idx: int,
) -> Dict[str, float]:
    """All metrics for one building."""
    return {
        "linear_dist_error": linear_distance_error(footprint, true_pt, estimated_pt),
        "euclidean_dist_error": euclidean_distance_error(true_pt, estimated_pt),
        **ranking_metrics(proba, true_pos_idx),
    }


def aggregate_results(results: List[Dict[str, float]]) -> Dict[str, float]:
    """Summary statistics over all buildings."""
    agg = {}
    for key in ["linear_dist_error", "euclidean_dist_error", "absolute_rank", "relative_rank"]:
        vals = np.array([r[key] for r in results])
        agg[f"mean_{key}"] = float(np.mean(vals))
        agg[f"median_{key}"] = float(np.median(vals))
        agg[f"std_{key}"] = float(np.std(vals))
        for pct in (80, 90, 95):
            agg[f"p{pct}_{key}"] = float(np.percentile(vals, pct))

    # CDF-style distance thresholds
    lin = np.array([r["linear_dist_error"] for r in results])
    for thresh in [0, 3, 5, 10, 20, 30, 50, 60, 100]:
        label = "exact" if thresh == 0 else f"{thresh}m"
        cutoff = 1.5 if thresh == 0 else thresh  # within one segment = exact
        agg[f"pct_within_{label}"] = float(np.mean(lin <= cutoff))

    # Ranking thresholds
    rel = np.array([r["relative_rank"] for r in results])
    for pct in [0.02, 0.05, 0.10, 0.20]:
        agg[f"pct_rank_top_{int(pct*100)}pct"] = float(np.mean(rel <= pct))

    return agg