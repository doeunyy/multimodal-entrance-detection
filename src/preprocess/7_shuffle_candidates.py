"""Shuffle the candidate entrance points for each building.

Step 6 stores candidates as positives followed by negatives, so the ordering
of "all_points"/"labels" leaks the label. This step randomly permutes each
building's candidates (keeping points and labels aligned) so downstream models
cannot exploit position.

Expected config keys in config.yaml:
- processed_data: path to processed data root
- cities: list of city names

Input/Output: Updates the "candidates" column of metadata.gpkg in-place. The
"positive" and "negative" entries are left unchanged; only "all_points" and
"labels" are shuffled (with a matched permutation).
"""

import hashlib
import json
import os
import random

import geopandas as gpd
import yaml
from tqdm import tqdm


def shuffle_candidates_payload(payload_str, seed):
    payload = json.loads(payload_str)
    all_points = payload.get("all_points", [])
    labels = payload.get("labels", [])

    if len(all_points) != len(labels):
        raise ValueError(
            f"all_points ({len(all_points)}) and labels ({len(labels)}) length mismatch"
        )

    rng = random.Random(seed)
    order = list(range(len(all_points)))
    rng.shuffle(order)

    payload["all_points"] = [all_points[i] for i in order]
    payload["labels"] = [labels[i] for i in order]
    return json.dumps(payload)


def shuffle_candidates_one_city(city):
    print(f"Processing city: {city}")
    metadata_path = os.path.join(processed_path, city, "metadata.gpkg")

    if not os.path.exists(metadata_path):
        print(f"  Warning: metadata not found at {metadata_path}")
        return

    gdf = gpd.read_file(metadata_path, layer="Filtered_Buildings")

    if "candidates" not in gdf.columns:
        raise ValueError(
            f"Required column 'candidates' not found in metadata layer for {city}. "
            "Run 6_add_candidates.py first."
        )

    shuffled = []
    for idx, payload_str in tqdm(
        enumerate(gdf["candidates"]), total=len(gdf), desc=f"  Shuffling {city}"
    ):
        seed = int.from_bytes(
            hashlib.md5(f"shuffle:{city}:{idx}".encode()).digest()[:4], "little"
        )
        shuffled.append(shuffle_candidates_payload(payload_str, seed))

    gdf["candidates"] = shuffled

    gdf.to_file(metadata_path, layer="Filtered_Buildings", driver="GPKG")
    print(f"  Updated metadata: {metadata_path}\n")


def shuffle_candidates(cities):
    for city in cities:
        shuffle_candidates_one_city(city)


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    processed_path = config["processed_data"]
    cities = config["cities"]

    shuffle_candidates(cities)
