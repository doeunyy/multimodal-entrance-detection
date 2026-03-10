#!/usr/bin/env bash
set -euo pipefail

python src/preprocess/1_filter_buildings_with_entrances.py
python src/preprocess/2_add_aerial_images.py
python src/preprocess/3_add_streetview_images.py
python src/preprocess/4_add_gps_traces.py
python src/preprocess/5_split_train_val_test.py
python src/preprocess/6_add_candidates.py
