"""
Preprocessing steps:
  1. Discretise building footprint perimeter into equidistant sample points
  2. Label the sample nearest to the true entrance as positive
  3. Select 'strong' negative samples for training
  4. Straw-man imputation for missing feature values
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, Polygon

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Footprint discretisation
# -----------------------------------------------------------------------

def discretize_footprint(
    footprint: Polygon, interval: float = 3.0,
) -> Tuple[List[Point], List[int], List[LineString]]:
    """
    Split footprint exterior ring into segments of ~`interval` metres.

    Returns
    -------
    samples : list[Point]
        Midpoint of each segment (candidate entrance locations).
    edge_indices : list[int]
        Index of the original polygon edge that contains each sample.
    master_edges : list[LineString]
        The original edges of the exterior ring.
    """
    coords = list(footprint.exterior.coords)  # last == first
    master_edges: List[LineString] = []
    for i in range(len(coords) - 1):
        master_edges.append(LineString([coords[i], coords[i + 1]]))

    samples: List[Point] = []
    edge_indices: List[int] = []

    for eidx, edge in enumerate(master_edges):
        elen = edge.length
        if elen < 1e-6:
            continue
        if elen <= interval:
            samples.append(edge.interpolate(0.5, normalized=True))
            edge_indices.append(eidx)
        else:
            n_seg = int(np.ceil(elen / interval))
            seg_len = elen / n_seg
            for j in range(n_seg):
                samples.append(edge.interpolate((j + 0.5) * seg_len))
                edge_indices.append(eidx)

    return samples, edge_indices, master_edges


# -----------------------------------------------------------------------
# Labelling
# -----------------------------------------------------------------------

def label_samples(samples: List[Point], true_entrance: Point) -> np.ndarray:
    """Mark the sample closest to the true entrance as positive (1)."""
    dists = np.array([s.distance(true_entrance) for s in samples])
    labels = np.zeros(len(samples), dtype=int)
    labels[int(np.argmin(dists))] = 1
    return labels


# -----------------------------------------------------------------------
# Strong negative selection
# -----------------------------------------------------------------------

def _ring_distance(ring: LineString, p1: Point, p2: Point) -> float:
    """Shortest distance along the ring between two points."""
    d1 = ring.project(p1)
    d2 = ring.project(p2)
    along = abs(d1 - d2)
    return min(along, ring.length - along)


def select_strong_negatives(
    samples: List[Point],
    labels: np.ndarray,
    features: np.ndarray,
    footprint: Polygon,
    pt_threshold: float = 24.0,
    ft_threshold: float = 0.04,
) -> np.ndarray:
    """
    Return boolean mask keeping all positives and 'strong' negatives.

    A negative is strong if BOTH:
      - physical distance along perimeter from positive >= PT
      - normalised feature-space distance from positive >= FT
    """
    ring = LineString(footprint.exterior.coords)
    pos_idx = int(np.where(labels == 1)[0][0])
    pos_pt = samples[pos_idx]

    # Min-max normalise features for distance calculation
    fmin = np.nanmin(features, axis=0)
    fmax = np.nanmax(features, axis=0)
    denom = fmax - fmin
    denom[denom == 0] = 1.0
    normed = (features - fmin) / denom
    normed = np.nan_to_num(normed, nan=0.0)
    pos_feat = normed[pos_idx]

    keep = np.zeros(len(samples), dtype=bool)
    keep[pos_idx] = True

    for i in range(len(samples)):
        if labels[i] == 1:
            continue
        phys = _ring_distance(ring, pos_pt, samples[i])
        feat = np.linalg.norm(normed[i] - pos_feat)
        if phys >= pt_threshold and feat >= ft_threshold:
            keep[i] = True

    return keep


# -----------------------------------------------------------------------
# Straw-man imputation
# -----------------------------------------------------------------------

def strawman_impute(
    df: pd.DataFrame,
    ref_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Fill missing values: median for numeric columns, mode for categorical.

    If ref_df is given (training set), compute fill values from it.
    Returns (imputed_df, fill_values_dict).
    """
    ref = ref_df if ref_df is not None else df
    fill: Dict = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            fill[col] = float(ref[col].median()) if not ref[col].isna().all() else 0.0
        else:
            mode = ref[col].mode()
            fill[col] = mode.iloc[0] if len(mode) > 0 else ""
    return df.fillna(fill), fill