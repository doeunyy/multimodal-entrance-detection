#!/usr/bin/env python3
"""
Main entry point for the building entrance tagging baseline.

Usage:
    python run.py --data_dir data --output_dir outputs

Expects GeoPackage files (*.gpkg) in data_dir, each with a
'Buildings_With_Entrances' layer.
"""
import argparse
import logging
import os
from pathlib import Path

import pandas as pd

from config import DataConfig, ModelConfig, PreprocessConfig
from data_loading import load_all_cities
from pipeline import build_dataset, run_cv, summarise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Building entrance tagging (Hu et al. 2020 baseline)",
    )
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--segment_interval", type=float, default=3.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Use explicit paths from config; override with data_dir if provided
    data_cfg = DataConfig()
    if args.data_dir != "data":
        # If user passed a custom data_dir, discover gpkg files recursively
        gpkg_files = sorted(Path(args.data_dir).rglob("*.gpkg"))
        if not gpkg_files:
            log.error("No .gpkg files found in %s (searched recursively)", args.data_dir)
            return
        data_cfg.gpkg_paths = [str(p) for p in gpkg_files]

    # Verify all paths exist
    missing = [p for p in data_cfg.gpkg_paths if not Path(p).exists()]
    if missing:
        log.error("Missing GeoPackage files: %s", missing)
        return
    log.info("Found %d GeoPackage(s): %s", len(data_cfg.gpkg_paths), data_cfg.gpkg_paths)
    prep_cfg = PreprocessConfig(segment_interval=args.segment_interval)
    model_cfg = ModelConfig(n_folds=args.n_folds)

    # --- Load ---
    log.info("=== Loading data ===")
    city_gdfs = load_all_cities(data_cfg)

    # --- Features ---
    log.info("=== Extracting features ===")
    dataset = build_dataset(city_gdfs, prep_cfg)
    feat_path = os.path.join(args.output_dir, "features.csv")
    dataset.to_csv(feat_path, index=False)
    log.info(
        "Samples: %d  |  Positive: %d  |  Negative: %d  |  Ratio 1:%d",
        len(dataset), int(dataset["label"].sum()),
        int((dataset["label"] == 0).sum()),
        int((dataset["label"] == 0).sum() / max(dataset["label"].sum(), 1)),
    )

    # --- Cross-validation ---
    log.info("=== Cross-validation ===")
    results = run_cv(dataset, model_cfg, prep_cfg)

    # --- Feature importance ---
    log.info("=== Feature importance ===")
    from pipeline import compute_feature_importance
    feat_imp = compute_feature_importance(dataset, model_cfg)
    feat_imp.to_csv(os.path.join(args.output_dir, "feature_importance.csv"), index=False)
    log.info("Top 10 features:\n%s", feat_imp.head(10).to_string(index=False))

    # --- Summary ---
    log.info("=== Results ===")
    summary = summarise(results)

    # Print overall results
    overall = summary[summary["city"] == "all"]
    log.info("Overall results:\n%s", overall.to_string(index=False))

    # Print per-city results
    per_city = summary[summary["city"] != "all"]
    if not per_city.empty:
        log.info("Per-city results:\n%s",
                 per_city[["model", "city", "n_buildings",
                           "mean_linear_dist_error", "median_linear_dist_error",
                           "pct_within_exact", "pct_within_10m", "pct_within_30m"]
                 ].to_string(index=False))

    summary.to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)

    for mname, rlist in results.items():
        if rlist:
            pd.DataFrame(rlist).to_csv(
                os.path.join(args.output_dir, "results_%s.csv" % mname),
                index=False,
            )

    log.info("Done. Outputs in %s/", args.output_dir)


if __name__ == "__main__":
    main()