"""Evaluate entrance detection predictions against labels.

Reads the labels created by 6_add_candidates.py and model predictions,
then computes evaluation metrics pooled across all candidates:
accuracy, precision, recall, F1, AUROC, AUPR.

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
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm


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

    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    # Baseline threshold=0.5 so accuracy/precision/recall are populated even when
    # no threshold yields a positive F1 (e.g. y_true is all zeros).
    best_threshold = 0.5
    baseline_pred = (y_pred_proba >= best_threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, baseline_pred)),
        "precision": float(precision_score(y_true, baseline_pred, zero_division=0)),
        "recall": float(recall_score(y_true, baseline_pred, zero_division=0)),
        "f1": float(f1_score(y_true, baseline_pred, zero_division=0)),
        "auroc": None,
        "aupr": None,
    }

    # AUROC / AUPR require both classes present in y_true.
    if len(np.unique(y_true)) > 1:
        try:
            metrics["auroc"] = float(roc_auc_score(y_true, y_pred_proba))
        except ValueError:
            metrics["auroc"] = None
        try:
            metrics["aupr"] = float(average_precision_score(y_true, y_pred_proba))
        except ValueError:
            metrics["aupr"] = None

        # Use precision_recall_curve to find the best-F1 threshold efficiently
        # (O(n log n) single-pass instead of a 101-point grid sweep).
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
        # precisions/recalls have one extra trailing entry; thresholds has len N-1.
        with np.errstate(invalid="ignore", divide="ignore"):
            pr_f1s = np.where(
                (precisions[:-1] + recalls[:-1]) > 0,
                2.0 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
                0.0,
            )
        best_idx = int(np.argmax(pr_f1s))
        best_pr_f1 = float(pr_f1s[best_idx])
        if best_pr_f1 > metrics["f1"]:
            best_threshold = float(thresholds[best_idx])
            metrics["f1"] = best_pr_f1
            y_pred_best = (y_pred_proba >= best_threshold).astype(int)
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred_best))
            metrics["precision"] = float(
                precision_score(y_true, y_pred_best, zero_division=0)
            )
            metrics["recall"] = float(recall_score(y_true, y_pred_best, zero_division=0))

    metrics["best_threshold"] = best_threshold

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


def parse_candidates_labels_only(value):
    """Parse only the labels from a serialized candidates payload.

    Skips WKT geometry decoding for performance when geometry is not needed.

    Args:
        value: Serialized candidates payload (JSON string or dict) from the
               metadata GeoPackage ``candidates`` column.

    Returns:
        List of integer labels (0/1) for each candidate. Returns an empty list
        if the payload is absent, malformed, or contains no ``labels`` key.
    """
    if value is None:
        return []

    payload = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return []

    if not isinstance(payload, dict):
        return []

    return list(payload.get("labels", []))


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
        "y_true": [],  # Pooled true labels across all buildings
        "y_pred_proba": [],  # Pooled predicted probabilities across all buildings
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

        candidates = parse_candidates_labels_only(row.candidates)
        y_true = np.array(candidates)

        # Verify prediction count matches candidate count
        if len(y_pred_proba) != len(y_true):
            print(
                f"    Warning: prediction count mismatch for {city}/building_{building_idx}: "
                f"{len(y_pred_proba)} predictions vs {len(y_true)} candidates"
            )
            continue

        building_metrics = {
            "n_candidates": len(y_true),
            "n_positive": int(np.sum(y_true)),
            "n_negative": int(len(y_true) - np.sum(y_true)),
        }

        city_results["buildings"][f"building_{building_idx}"] = building_metrics
        city_results["y_true"].extend(y_true.tolist())
        city_results["y_pred_proba"].extend(np.asarray(y_pred_proba).tolist())

    return city_results


def aggregate_metrics(all_city_results):
    """Aggregate metrics across all cities.

    Args:
        all_city_results: Dictionary of city -> results

    Returns:
        Dictionary with aggregated metrics
    """
    total_buildings = 0
    total_candidates = 0
    total_positive = 0
    total_negative = 0
    pooled_y_true: list[int] = []
    pooled_y_pred_proba: list[float] = []

    for _, results in all_city_results.items():
        total_buildings += results["total_buildings"]
        pooled_y_true.extend(results.get("y_true", []))
        pooled_y_pred_proba.extend(results.get("y_pred_proba", []))

        for _, metrics in results["buildings"].items():
            total_candidates += metrics.get("n_candidates", 0)
            total_positive += metrics.get("n_positive", 0)
            total_negative += metrics.get("n_negative", 0)

    aggregated_summary: dict[str, object] = {
        "total_buildings": total_buildings,
        "total_candidates": total_candidates,
        "total_positive": total_positive,
        "total_negative": total_negative,
        "pooled_metrics": {},
    }

    # Pool all candidates across buildings and compute metrics once.
    if pooled_y_true:
        aggregated_summary["pooled_metrics"] = compute_candidate_metrics(
            np.array(pooled_y_true), np.array(pooled_y_pred_proba)
        )

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

    print("\nPooled Metrics (all candidates across all buildings):")
    pooled = aggregated.get("pooled_metrics") or {}
    for metric_name, value in pooled.items():
        if value is None:
            print(f"  {metric_name}: N/A")
        else:
            print(f"  {metric_name}: {value:.4f}")

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
