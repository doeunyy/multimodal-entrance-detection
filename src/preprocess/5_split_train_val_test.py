"""Split buildings into train, validation, and test sets.

Reads metadata.gpkg files from each city, assigns buildings to splits,
and saves a JSON file mapping building identifiers to their split assignment.

Expected config keys in config.yaml:
- processed_data: path to processed data root
- cities: list of city names

Output: {processed_data}/splits.json with structure:
{
    "nyc": {
        "building_0": "train",
        "building_1": "val",
        ...
    },
    "la": {...},
    ...
}
"""

import json
import os
import random

import geopandas as gpd
import yaml

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def split_buildings_one_city(city, gdf):
    """Split buildings from one city into train/val/test.

    Args:
        city: City name
        gdf: GeoDataFrame with buildings

    Returns:
        Dictionary mapping building indices to split names
    """
    n_buildings = len(gdf)
    indices = list(range(n_buildings))

    # Shuffle with fixed seed for reproducibility
    random.seed(RANDOM_SEED)
    random.shuffle(indices)

    # Calculate split sizes
    n_train = int(n_buildings * TRAIN_RATIO)
    n_val = int(n_buildings * VAL_RATIO)

    # Assign splits
    splits = {}
    for i, idx in enumerate(indices):
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"
        splits[f"building_{idx}"] = split

    # Count buildings per split
    n_train_actual = sum(1 for s in splits.values() if s == "train")
    n_val_actual = sum(1 for s in splits.values() if s == "val")
    n_test_actual = sum(1 for s in splits.values() if s == "test")

    print(f"  {city}: {n_buildings} buildings")
    print(f"    Train: {n_train_actual} ({n_train_actual/n_buildings*100:.1f}%)")
    print(f"    Val:   {n_val_actual} ({n_val_actual/n_buildings*100:.1f}%)")
    print(f"    Test:  {n_test_actual} ({n_test_actual/n_buildings*100:.1f}%)")

    return splits


def create_splits(cities):
    """Create train/val/test splits for all cities."""
    all_splits = {}

    for city in cities:
        print(f"Processing city: {city}")
        metadata_path = os.path.join(processed_path, city, "metadata.gpkg")

        if not os.path.exists(metadata_path):
            print(f"  Warning: metadata not found at {metadata_path}")
            continue

        gdf = gpd.read_file(metadata_path, layer="Filtered_Buildings")
        city_splits = split_buildings_one_city(city, gdf)
        all_splits[city] = city_splits

    # Save splits to JSON file
    output_path = os.path.join(processed_path, "splits.json")
    with open(output_path, "w") as f:
        json.dump(all_splits, f, indent=2)

    print(f"\nSaved splits to: {output_path}")

    # Print overall statistics
    total_train = sum(
        1
        for city_splits in all_splits.values()
        for split in city_splits.values()
        if split == "train"
    )
    total_val = sum(
        1
        for city_splits in all_splits.values()
        for split in city_splits.values()
        if split == "val"
    )
    total_test = sum(
        1
        for city_splits in all_splits.values()
        for split in city_splits.values()
        if split == "test"
    )
    total = total_train + total_val + total_test

    print(f"\nOverall statistics:")
    print(f"  Total: {total} buildings")
    print(f"  Train: {total_train} ({total_train/total*100:.1f}%)")
    print(f"  Val:   {total_val} ({total_val/total*100:.1f}%)")
    print(f"  Test:  {total_test} ({total_test/total*100:.1f}%)")


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    processed_path = config["processed_data"]
    cities = config["cities"]

    create_splits(cities)
