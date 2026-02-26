from typing import Dict, List, Tuple

import numpy as np
from shapely.geometry import Point


def map_to_candidates(
    raw_json: Dict, candidates: List[Tuple[float, float]], threshold_deg: float = 0.0001
) -> List[float]:
    """
    Maps LLM-generated coordinates to the closest candidate points.

    Args:
        raw_json: The 'entrances' list from Kimi API response.
        candidates: List of (lon, lat) tuples from the dataset schema.
        threshold_deg: Distance threshold (~10m in degrees) to consider a match.
    """
    entrances = raw_json.get("entrances", [])
    # Initialize probabilities for each candidate to 0.0
    probs = np.zeros(len(candidates))

    if not entrances or not candidates:
        return probs.tolist()

    for ent in entrances:
        try:
            # Ensure coordinates are floats
            p_lat, p_lon = float(ent["lat"]), float(ent["lon"])
            conf = float(ent.get("confidence", 1.0))
            pred_pt = Point(p_lon, p_lat)

            # Calculate distance to all candidate (lon, lat) tuples
            # Schema defines candidate as (lon, lat)
            distances = [pred_pt.distance(Point(c[0], c[1])) for c in candidates]
            closest_idx = np.argmin(distances)

            # Assign confidence if the prediction is within the spatial threshold
            if distances[closest_idx] < threshold_deg:
                probs[closest_idx] = max(probs[closest_idx], conf)
        except (KeyError, ValueError, TypeError):
            continue

    return probs.tolist()
