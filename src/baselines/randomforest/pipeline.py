"""
End-to-end pipeline.

1. Load all city GeoPackages
2. For each building: discretise footprint, extract features, label samples
3. 5-fold cross-validation: strong-negative selection, imputation, train, tag
4. Compute and aggregate evaluation metrics
"""
import logging
import os
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import MultiPolygon, Point, Polygon
from sklearn.model_selection import KFold

from config import DataConfig, ModelConfig, PreprocessConfig
from data_loading import load_all_cities
from data_preprocessing import (
    discretize_footprint,
    label_samples,
    select_strong_negatives,
    strawman_impute,
)
from feature_extraction import extract_features_for_building
from models import build_all_models, predict_entrance_proba
from evaluation import aggregate_results, compute_building_metrics

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Feature columns helper
# -----------------------------------------------------------------------

METADATA_COLS = {
    "building_id", "label", "sample_x", "sample_y",
    "footprint_wkt", "entrance_x", "entrance_y", "city",
}


def _feat_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in METADATA_COLS]


# -----------------------------------------------------------------------
# Step 1-2: Build the full sample-level dataset
# -----------------------------------------------------------------------

def build_dataset(
    city_gdfs: Dict[str, gpd.GeoDataFrame],
    prep_cfg: PreprocessConfig,
) -> pd.DataFrame:
    """
    For every building in every city: discretise, extract features, label.

    Returns a single DataFrame (all samples across all buildings).
    """
    frames: List[pd.DataFrame] = []

    for city, city_gdf in city_gdfs.items():
        for _, row in city_gdf.iterrows():
            bid = row["building_id"]
            fp = row.geometry
            entrance = row["entrance_pt"]

            if isinstance(fp, MultiPolygon):
                fp = max(fp.geoms, key=lambda g: g.area)

            samples, edge_indices, master_edges = discretize_footprint(
                fp, interval=prep_cfg.segment_interval,
            )
            if len(samples) < 3:
                continue

            labels = label_samples(samples, entrance)

            # Nearby buildings for inter-building features
            nearby = []
            centroid = fp.centroid
            for _, other in city_gdf.iterrows():
                if other["building_id"] == bid:
                    continue
                og = other.geometry
                if og is not None and centroid.distance(og) < 200:
                    nearby.append(og)

            has_addr = bool(row.get("addr_street"))

            feat_df = extract_features_for_building(
                samples, edge_indices, master_edges, fp, nearby, has_addr,
            )

            # Attach metadata
            feat_df["building_id"] = bid
            feat_df["label"] = labels
            feat_df["sample_x"] = [p.x for p in samples]
            feat_df["sample_y"] = [p.y for p in samples]
            feat_df["footprint_wkt"] = fp.wkt
            feat_df["entrance_x"] = entrance.x
            feat_df["entrance_y"] = entrance.y
            feat_df["city"] = city

            frames.append(feat_df)

            if len(frames) % 100 == 0:
                log.info("  Processed %d buildings...", len(frames))

    log.info("Feature extraction complete: %d buildings", len(frames))
    return pd.concat(frames, ignore_index=True)


# -----------------------------------------------------------------------
# Step 3-4: Cross-validation
# -----------------------------------------------------------------------

def run_cv(
    dataset: pd.DataFrame,
    model_cfg: ModelConfig,
    prep_cfg: PreprocessConfig,
) -> Dict[str, List[Dict]]:
    """
    5-fold cross-validation over buildings.

    Returns: {model_name: [per-building metrics dict, ...]}.
    """
    bids = dataset["building_id"].unique()
    kf = KFold(
        n_splits=model_cfg.n_folds, shuffle=True,
        random_state=model_cfg.random_state,
    )
    feat_cols = _feat_cols(dataset)
    results: Dict[str, List[Dict]] = {
        name: [] for name in ["WRF", "BRF", "SmoteBoost", "RF"]
    }

    for fold, (train_idx, test_idx) in enumerate(kf.split(bids)):
        train_bids = set(bids[train_idx])
        test_bids = set(bids[test_idx])
        log.info(
            "Fold %d/%d: %d train, %d test buildings",
            fold + 1, model_cfg.n_folds, len(train_bids), len(test_bids),
        )

        train_df = dataset[dataset["building_id"].isin(train_bids)].copy()
        test_df = dataset[dataset["building_id"].isin(test_bids)].copy()

        # --- Strong negative selection on training set ---
        keep = np.ones(len(train_df), dtype=bool)
        for bid, grp in train_df.groupby("building_id"):
            idx = grp.index.tolist()
            labs = grp["label"].values
            if labs.sum() == 0:
                continue
            feats = grp[feat_cols].fillna(0).values
            fp = wkt.loads(grp["footprint_wkt"].iloc[0])
            pts = [
                Point(x, y)
                for x, y in zip(grp["sample_x"], grp["sample_y"])
            ]
            local_keep = select_strong_negatives(
                pts, labs, feats, fp,
                pt_threshold=prep_cfg.physical_distance_threshold,
                ft_threshold=prep_cfg.feature_distance_threshold,
            )
            for li, gi in enumerate(idx):
                if not local_keep[li]:
                    keep[train_df.index.get_loc(gi)] = False

        train_df = train_df[keep]

        # --- Imputation ---
        train_feat, fill = strawman_impute(train_df[feat_cols])
        test_feat = test_df[feat_cols].fillna(fill)

        X_train = np.nan_to_num(train_feat.values.astype(float), nan=0.0)
        y_train = train_df["label"].values
        X_test = np.nan_to_num(test_feat.values.astype(float), nan=0.0)

        # --- Train each model and tag test buildings ---
        models = build_all_models(model_cfg)
        for mname, model in models.items():
            log.info("  Training %s...", mname)
            try:
                model.fit(X_train, y_train)
            except Exception as e:
                log.warning("  %s failed: %s", mname, e)
                continue

            for bid in test_bids:
                mask = test_df["building_id"] == bid
                X_bid = X_test[mask.values]
                if len(X_bid) == 0:
                    continue

                proba = predict_entrance_proba(model, X_bid)
                est_idx = int(np.argmax(proba))

                bid_rows = test_df[mask]
                est_pt = Point(
                    bid_rows["sample_x"].iloc[est_idx],
                    bid_rows["sample_y"].iloc[est_idx],
                )
                true_pt = Point(
                    bid_rows["entrance_x"].iloc[0],
                    bid_rows["entrance_y"].iloc[0],
                )
                fp = wkt.loads(bid_rows["footprint_wkt"].iloc[0])

                labs = bid_rows["label"].values
                tp_idx = int(np.where(labs == 1)[0][0]) if labs.sum() > 0 else 0

                m = compute_building_metrics(fp, true_pt, est_pt, proba, tp_idx)
                m["building_id"] = bid
                m["fold"] = fold
                m["city"] = bid_rows["city"].iloc[0] if "city" in bid_rows.columns else "unknown"
                results[mname].append(m)

    return results


# -----------------------------------------------------------------------
# Step 5: Summary
# -----------------------------------------------------------------------

def summarise(results: Dict[str, List[Dict]]) -> pd.DataFrame:
    rows = []
    for mname, rlist in results.items():
        if not rlist:
            continue
        # Overall
        agg = aggregate_results(rlist)
        agg["model"] = mname
        agg["city"] = "all"
        agg["n_buildings"] = len(rlist)
        rows.append(agg)

        # Per-city breakdown
        cities = set(r.get("city", "unknown") for r in rlist)
        for city in sorted(cities):
            city_results = [r for r in rlist if r.get("city") == city]
            if city_results:
                city_agg = aggregate_results(city_results)
                city_agg["model"] = mname
                city_agg["city"] = city
                city_agg["n_buildings"] = len(city_results)
                rows.append(city_agg)

    return pd.DataFrame(rows)


def compute_feature_importance(
    dataset: pd.DataFrame, model_cfg,
) -> pd.DataFrame:
    """
    Train a single WRF on all data and extract feature importances.
    Returns DataFrame sorted by importance.
    """
    from models import build_wrf
    from data_preprocessing import strawman_impute

    feat_cols = [c for c in dataset.columns if c not in METADATA_COLS]
    X, _ = strawman_impute(dataset[feat_cols])
    X = np.nan_to_num(X.values.astype(float), nan=0.0)
    y = dataset["label"].values

    model = build_wrf(model_cfg)
    model.fit(X, y)

    imp = pd.DataFrame({
        "feature": feat_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    imp["importance_pct"] = imp["importance"] / imp["importance"].sum() * 100
    return imp