"""Evaluate entrance detection predictions against labels.

Reads the labels created by 6_add_candidates.py and model predictions,
then computes comprehensive evaluation metrics:
- Per-candidate: accuracy, F1, Recall, Precision, AUROC
- Per-building aggregated scores of the above
- Aggregated across true entrances: positive probability given to true entrance
- Aggregated across buildings: positive probability given to true entrance

Expected input:
- metadata.gpkg: Contains candidates column with positive/negative labels
- predictions file: CSV or similar with predicted probabilities for each candidate
  Format: city,building_idx,candidate_idx,predicted_prob

Output: A comprehensive metrics report saved to a JSON file
"""

import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from shapely import wkt
from shapely.geometry import Point
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm


def project_point_to_boundary(point, geometry):
    """Project a point onto the building boundary.

    If the point is already on the boundary, return it. Otherwise, find the
    closest point on the boundary.

    Args:
        point: Shapely Point
        geometry: Shapely Polygon or MultiPolygon

    Returns:
        Shapely Point on the boundary
    """
    boundary = geometry.boundary

    # If point is already on boundary, return it
    if boundary.distance(point) < 1e-9:
        return point

    # Find closest point on boundary
    nearest = boundary.interpolate(boundary.project(point))
    return nearest


def _ring_distance(d1, d2, ring_length):
    """Return the minimum arc-length distance along a closed ring.

    Args:
        d1: Float distance of point 1 along the ring
        d2: Float distance of point 2 along the ring
        ring_length: Total length of the ring

    Returns:
        Float minimum arc-length distance
    """
    direct = abs(d1 - d2)
    return min(direct, ring_length - direct)


def distance_along_perimeter(point1, point2, geometry):
    """Compute distance along the building perimeter between two points.

    Both points are projected onto the boundary. The distance is computed as
    the minimum arc length along the perimeter between them.

    For ``MultiPolygon`` buildings the boundary consists of multiple
    disconnected rings.  Each point is first assigned to its nearest polygon
    component by boundary distance.  If both points are nearest to the same
    component the ring distance is computed on that component alone.  When
    the points fall on different components a cross-ring distance is not
    well-defined and ``None`` is returned.

    Args:
        point1: Shapely Point
        point2: Shapely Point
        geometry: Shapely Polygon or MultiPolygon

    Returns:
        Float distance in same units as geometry, or ``None`` when the two
        points belong to different components of a ``MultiPolygon``.
    """
    # MultiPolygon: assign each point to its nearest component boundary, then
    # compute the ring distance only when both points share the same component.
    if hasattr(geometry, "geoms"):
        polys = list(geometry.geoms)
        boundaries = [poly.boundary for poly in polys]
        # Compute boundary distances once per point in a single pass
        dists1 = [b.distance(point1) for b in boundaries]
        dists2 = [b.distance(point2) for b in boundaries]
        nearest_idx1 = dists1.index(min(dists1))
        nearest_idx2 = dists2.index(min(dists2))

        if nearest_idx1 != nearest_idx2:
            # Points project onto different disconnected rings: distance is
            # not well-defined for a single perimeter arc.
            return None

        return distance_along_perimeter(point1, point2, polys[nearest_idx1])

    # Single Polygon: project both points to its boundary (a LinearRing)
    boundary = geometry.boundary
    p1_proj = project_point_to_boundary(point1, geometry)
    p2_proj = project_point_to_boundary(point2, geometry)

    d1 = boundary.project(p1_proj)
    d2 = boundary.project(p2_proj)

    return _ring_distance(d1, d2, boundary.length)


def euclidean_distance(point1, point2):
    """Compute Euclidean distance between two points.

    Args:
        point1: Shapely Point
        point2: Shapely Point

    Returns:
        Float Euclidean distance
    """
    return point1.distance(point2)


def compute_weighted_distance_metrics(
    all_candidates, y_pred_proba, y_true, building_geom
):
    """Compute probability-weighted distances to closest true entrance.

    For each candidate, find the distance to the closest true entrance,
    weight it by the predicted probability, and compute aggregate statistics.

    Args:
        all_candidates: List of Point geometries (all candidates)
        y_pred_proba: Array of predicted probabilities
        y_true: Array of true labels (0/1)
        building_geom: Shapely Polygon or MultiPolygon

    Returns:
        Dictionary with distance-based metrics
    """
    metrics = {}

    # Get true entrance points (positive samples)
    true_entrance_indices = np.where(y_true == 1)[0]
    if len(true_entrance_indices) == 0:
        # No true entrances, can't compute distance metrics
        metrics["perimeter_weighted_distances"] = None
        metrics["euclidean_weighted_distances"] = None
        return metrics

    true_entrances = [all_candidates[idx] for idx in true_entrance_indices]

    # Compute weighted distances for each candidate
    perimeter_weighted_dists = []
    euclidean_weighted_dists = []

    for candidate_idx, candidate_point in enumerate(all_candidates):
        pred_prob = y_pred_proba[candidate_idx]

        # Find distance to closest true entrance
        perimeter_dists = [
            distance_along_perimeter(candidate_point, entrance, building_geom)
            for entrance in true_entrances
        ]
        # Filter out None values (cross-component MultiPolygon pairs)
        perimeter_dists_valid = [d for d in perimeter_dists if d is not None]
        euclidean_dists = [
            euclidean_distance(candidate_point, entrance) for entrance in true_entrances
        ]

        min_euclidean_dist = min(euclidean_dists)

        # Weight by predicted probability.
        # When all perimeter distances are None (cross-component), skip this
        # candidate; `perimeter_weighted_dists` may be shorter than the full
        # candidate list, and the caller handles the empty-list case by setting
        # `perimeter_weighted_distances` to None in the metrics dict.
        if perimeter_dists_valid:
            perimeter_weighted_dists.append(min(perimeter_dists_valid) * pred_prob)
        euclidean_weighted_dists.append(min_euclidean_dist * pred_prob)

    # Compute statistics
    if perimeter_weighted_dists:
        perimeter_weighted_dists = np.array(perimeter_weighted_dists)
        metrics["perimeter_weighted_distances"] = {
            "mean": float(np.mean(perimeter_weighted_dists)),
            "std": float(np.std(perimeter_weighted_dists)),
            "min": float(np.min(perimeter_weighted_dists)),
            "max": float(np.max(perimeter_weighted_dists)),
            "median": float(np.median(perimeter_weighted_dists)),
        }
    else:
        metrics["perimeter_weighted_distances"] = None

    if euclidean_weighted_dists:
        euclidean_weighted_dists = np.array(euclidean_weighted_dists)
        metrics["euclidean_weighted_distances"] = {
            "mean": float(np.mean(euclidean_weighted_dists)),
            "std": float(np.std(euclidean_weighted_dists)),
            "min": float(np.min(euclidean_weighted_dists)),
            "max": float(np.max(euclidean_weighted_dists)),
            "median": float(np.median(euclidean_weighted_dists)),
        }
    else:
        metrics["euclidean_weighted_distances"] = None

    return metrics


def load_predictions(predictions_file):
    """Load predictions from file.

    Expected format: CSV with columns:
    - city, building_idx, candidate_idx, predicted_prob

    Returns:
        Dictionary mapping (city, building_idx) -> array of predictions
    """
    df = pd.read_csv(predictions_file)
    predictions = {}

    for (city, building_idx), group in df.groupby(["city", "building_idx"]):
        # Sort by candidate_idx to maintain order
        group = group.sort_values("candidate_idx")
        predictions[(city, building_idx)] = group["predicted_prob"].values

    return predictions


def compute_candidate_metrics(y_true, y_pred_proba):
    """Compute per-candidate metrics.

    Args:
        y_true: Array of true labels (0/1)
        y_pred_proba: Array of predicted probabilities

    Returns:
        Dictionary with metrics
    """
    # Convert probabilities to binary predictions at threshold 0.5
    y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # AUROC only if there's variation in predictions
    if len(np.unique(y_pred_proba)) > 1:
        try:
            metrics["auroc"] = float(roc_auc_score(y_true, y_pred_proba))
        except ValueError:
            metrics["auroc"] = None
    else:
        metrics["auroc"] = None

    return metrics


def parse_candidates(value):
    """Parse serialized candidates payload from metadata GeoPackage."""
    if value is None:
        return {}

    payload = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return {}

    if not isinstance(payload, dict):
        return {}

    candidates = {
        "labels": list(payload.get("labels", [])),
        "all_points": [],
    }

    for item in payload.get("all_points", []):
        if isinstance(item, Point):
            candidates["all_points"].append(item)
            continue
        if not isinstance(item, str):
            continue
        try:
            geom = wkt.loads(item)
            if isinstance(geom, Point):
                candidates["all_points"].append(geom)
        except Exception:
            continue

    return candidates


def evaluate_one_city(city, gdf, predictions):
    """Evaluate all buildings in one city.

    Args:
        city: City name
        gdf: GeoDataFrame with candidates column
        predictions: Dictionary of (city, bidx) -> predictions array

    Returns:
        Dictionary with city-level results
    """
    city_results = {
        "total_buildings": len(gdf),
        "buildings": {},
        "true_entrance_probs": [],  # Probabilities assigned to true entrances
    }

    for building_idx, row in tqdm(
        enumerate(gdf.itertuples()),
        total=len(gdf),
        desc=f"  Evaluating {city}",
        leave=False,
    ):
        pred_key = (city, building_idx)

        if pred_key not in predictions:
            print(f"    Warning: no predictions for {city}/building_{building_idx}")
            continue

        y_pred_proba = predictions[pred_key]

        # Get true labels
        if not hasattr(row, "candidates") or not row.candidates:
            print(f"    Warning: no candidates for {city}/building_{building_idx}")
            continue

        candidates = parse_candidates(row.candidates)
        y_true = np.array(candidates.get("labels", []))
        all_candidates = candidates.get("all_points", [])

        # Verify prediction count matches candidate count
        if len(y_pred_proba) != len(y_true):
            print(
                f"    Warning: prediction count mismatch for {city}/building_{building_idx}: "
                f"{len(y_pred_proba)} predictions vs {len(y_true)} candidates"
            )
            continue

        # Per-candidate metrics
        building_metrics = compute_candidate_metrics(y_true, y_pred_proba)

        # Building-level aggregation: average the candidate-level metrics
        # (This represents overall building performance)
        building_metrics["n_candidates"] = len(y_true)
        building_metrics["n_positive"] = int(np.sum(y_true))
        building_metrics["n_negative"] = int(len(y_true) - np.sum(y_true))

        # Store probabilities assigned to true entrances
        positive_probs = y_pred_proba[y_true == 1]
        building_metrics["true_entrance_probs"] = positive_probs.tolist()
        building_metrics["mean_prob_for_true_entrance"] = (
            float(np.mean(positive_probs)) if len(positive_probs) > 0 else None
        )

        # Compute distance-based metrics if we have candidate geometries
        if all_candidates and len(all_candidates) == len(y_true):
            distance_metrics = compute_weighted_distance_metrics(
                all_candidates, y_pred_proba, y_true, row.geometry
            )
            building_metrics.update(distance_metrics)

        city_results["buildings"][f"building_{building_idx}"] = building_metrics
        city_results["true_entrance_probs"].extend(positive_probs.tolist())

    return city_results


def aggregate_metrics(all_city_results):
    """Aggregate metrics across all cities.

    Args:
        all_city_results: Dictionary of city -> results

    Returns:
        Dictionary with aggregated metrics
    """
    aggregated = {
        "total_cities": len(all_city_results),
        "total_buildings": 0,
        "total_candidates": 0,
        "total_positive": 0,
        "total_negative": 0,
        "per_building_avg_metrics": {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "auroc": [],
        },
        "all_true_entrance_probs": [],
        "perimeter_weighted_distances_all": [],
        "euclidean_weighted_distances_all": [],
    }

    for city, results in all_city_results.items():
        aggregated["total_buildings"] += results["total_buildings"]

        for building_id, metrics in results["buildings"].items():
            aggregated["total_candidates"] += metrics.get("n_candidates", 0)
            aggregated["total_positive"] += metrics.get("n_positive", 0)
            aggregated["total_negative"] += metrics.get("n_negative", 0)

            # Collect per-building metrics
            for metric_name in ["accuracy", "precision", "recall", "f1", "auroc"]:
                if metric_name in metrics and metrics[metric_name] is not None:
                    aggregated["per_building_avg_metrics"][metric_name].append(
                        metrics[metric_name]
                    )

            # Collect true entrance probabilities
            if "true_entrance_probs" in metrics:
                aggregated["all_true_entrance_probs"].extend(
                    metrics["true_entrance_probs"]
                )

            # Collect distance metrics
            if "perimeter_weighted_distances" in metrics:
                perim_metrics = metrics["perimeter_weighted_distances"]
                if perim_metrics is not None:
                    aggregated["perimeter_weighted_distances_all"].append(
                        perim_metrics["mean"]
                    )

            if "euclidean_weighted_distances" in metrics:
                eucl_metrics = metrics["euclidean_weighted_distances"]
                if eucl_metrics is not None:
                    aggregated["euclidean_weighted_distances_all"].append(
                        eucl_metrics["mean"]
                    )

    # Compute means of per-building metrics
    aggregated_summary = {
        "total_buildings": aggregated["total_buildings"],
        "total_candidates": aggregated["total_candidates"],
        "total_positive": aggregated["total_positive"],
        "total_negative": aggregated["total_negative"],
        "per_building_avg_metrics": {},
        "across_true_entrances": {},
        "distance_metrics": {},
    }

    for metric_name, values in aggregated["per_building_avg_metrics"].items():
        if len(values) > 0:
            aggregated_summary["per_building_avg_metrics"][metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        else:
            aggregated_summary["per_building_avg_metrics"][metric_name] = None

    # Statistics across true entrances
    if len(aggregated["all_true_entrance_probs"]) > 0:
        probs = np.array(aggregated["all_true_entrance_probs"])
        aggregated_summary["across_true_entrances"] = {
            "mean_prob": float(np.mean(probs)),
            "std_prob": float(np.std(probs)),
            "min_prob": float(np.min(probs)),
            "max_prob": float(np.max(probs)),
            "median_prob": float(np.median(probs)),
            "n_true_entrances": len(probs),
        }

    # Distance metrics aggregation
    if len(aggregated["perimeter_weighted_distances_all"]) > 0:
        perim_dists = np.array(aggregated["perimeter_weighted_distances_all"])
        aggregated_summary["distance_metrics"]["perimeter_weighted_distance"] = {
            "mean": float(np.mean(perim_dists)),
            "std": float(np.std(perim_dists)),
            "min": float(np.min(perim_dists)),
            "max": float(np.max(perim_dists)),
            "median": float(np.median(perim_dists)),
        }

    if len(aggregated["euclidean_weighted_distances_all"]) > 0:
        eucl_dists = np.array(aggregated["euclidean_weighted_distances_all"])
        aggregated_summary["distance_metrics"]["euclidean_weighted_distance"] = {
            "mean": float(np.mean(eucl_dists)),
            "std": float(np.std(eucl_dists)),
            "min": float(np.min(eucl_dists)),
            "max": float(np.max(eucl_dists)),
            "median": float(np.median(eucl_dists)),
        }

    return aggregated_summary


def evaluate_predictions(processed_path, predictions_file, cities, output_file):
    """Main evaluation function.

    Args:
        processed_path: Path to processed data root
        predictions_file: Path to predictions CSV file
        cities: List of city names
        output_file: Path to save results JSON
    """
    print("Loading predictions...")
    predictions = load_predictions(predictions_file)

    print("Evaluating predictions by city...")
    all_city_results = {}

    for city in cities:
        print(f"Processing city: {city}")
        metadata_path = os.path.join(processed_path, city, "metadata.gpkg")

        if not os.path.exists(metadata_path):
            print(f"  Warning: metadata not found at {metadata_path}")
            continue

        gdf = gpd.read_file(metadata_path, layer="Filtered_Buildings")
        city_results = evaluate_one_city(city, gdf, predictions)
        all_city_results[city] = city_results

        print(f"  Evaluated {len(city_results['buildings'])} buildings")

    # Aggregate results
    print("\nAggregating results...")
    aggregated = aggregate_metrics(all_city_results)

    # Prepare final output
    output = {
        "meta": {
            "predictions_file": predictions_file,
            "n_cities": len(cities),
            "cities": cities,
        },
        "per_city": all_city_results,
        "aggregated": aggregated,
    }

    # Save results
    print(f"\nSaving results to: {output_file}")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\nTotal buildings: {aggregated['total_buildings']}")
    print(f"Total candidates: {aggregated['total_candidates']}")
    print(f"Total positive samples: {aggregated['total_positive']}")
    print(f"Total negative samples: {aggregated['total_negative']}")

    print("\nPer-Building Average Metrics:")
    for metric_name, stats in aggregated["per_building_avg_metrics"].items():
        if stats:
            print(f"  {metric_name}:")
            print(
                f"    Mean: {stats['mean']:.4f} ± {stats['std']:.4f} "
                f"(range: [{stats['min']:.4f}, {stats['max']:.4f}])"
            )

    if aggregated["across_true_entrances"]:
        stats = aggregated["across_true_entrances"]
        print(f"\nAcross True Entrances (n={stats['n_true_entrances']}):")
        print(f"  Mean probability: {stats['mean_prob']:.4f} ± {stats['std_prob']:.4f}")
        print(f"  Median probability: {stats['median_prob']:.4f}")
        print(f"  Range: [{stats['min_prob']:.4f}, {stats['max_prob']:.4f}]")

    if aggregated.get("distance_metrics"):
        print("\nProbability-Weighted Distance Metrics:")

        if "perimeter_weighted_distance" in aggregated["distance_metrics"]:
            stats = aggregated["distance_metrics"]["perimeter_weighted_distance"]
            print(f"  Perimeter-based distance:")
            print(
                f"    Mean: {stats['mean']:.4f} ± {stats['std']:.4f} "
                f"(range: [{stats['min']:.4f}, {stats['max']:.4f}])"
            )
            print(f"    Median: {stats['median']:.4f}")

        if "euclidean_weighted_distance" in aggregated["distance_metrics"]:
            stats = aggregated["distance_metrics"]["euclidean_weighted_distance"]
            print(f"  Euclidean distance:")
            print(
                f"    Mean: {stats['mean']:.4f} ± {stats['std']:.4f} "
                f"(range: [{stats['min']:.4f}, {stats['max']:.4f}])"
            )
            print(f"    Median: {stats['median']:.4f}")

    print("=" * 60)

    return output


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    processed_path = config["processed_data"]
    cities = config["cities"]

    # Expected to receive predictions_file path from command line or config
    predictions_file = config.get(
        "predictions_file", os.path.join(processed_path, "predictions.csv")
    )

    output_file = config.get(
        "evaluation_output", os.path.join(processed_path, "evaluation_metrics.json")
    )

    evaluate_predictions(processed_path, predictions_file, cities, output_file)
