"""Data loader for multimodal building entrance detection.

Provides a PyTorch Dataset that loads buildings with their associated
multimodal data (streetview, aerial, GPS) and entrance labels.

Each instance is a building with its associated information.
The label is the set of entrance locations (lat/lon coordinates).
"""

import json
import os
from typing import Dict, List, Optional, Set

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
import yaml
from PIL import Image
from shapely import wkt
from shapely.geometry import Point
from torch.utils.data import Dataset


class BuildingEntranceDataset(Dataset):
    """Dataset for multimodal building entrance detection (binary classification).

    For each building, provides candidate entrance points and binary labels
    indicating whether each candidate is a true entrance (1) or not (0).

    Args:
        city: City name to load data from
        split: One of 'train', 'val', or 'test'
        modalities: List of modalities to include. Can contain 'streetview',
                   'aerial', and/or 'gps'. Default is all modalities.
        config_path: Path to config.yaml file. Default is 'config.yaml'

    Returns:
        Dictionary with keys:
        - 'building_id': (city, building_idx) tuple
        - 'geometry': Building footprint as shapely geometry
        - 'candidates': List of candidate entrance Points
        - 'labels': Binary labels for each candidate (1=true entrance, 0=negative)
        - 'streetview_images': List of PIL Images (if 'streetview' in modalities)
        - 'aerial_image': Numpy array (C, H, W) (if 'aerial' in modalities)
        - 'gps_traces': DataFrame of GPS points (if 'gps' in modalities)
    """

    def __init__(
        self,
        city: str,
        split: str = "train",
        modalities: Optional[List[str]] = None,
        config_path: str = "config.yaml",
    ):
        """Initialize the dataset.

        Args:
            city: City name to load data from
            split: One of 'train', 'val', or 'test'
            modalities: List of modalities to use. Default is all.
            config_path: Path to config.yaml
        """
        assert split in [
            "train",
            "val",
            "test",
        ], f"Split must be 'train', 'val', or 'test', got '{split}'"

        self.city = city
        self.split = split
        self.modalities = set(
            modalities if modalities else ["streetview", "aerial", "gps"]
        )

        # Validate modalities
        valid_modalities = {"streetview", "aerial", "gps"}
        invalid = self.modalities - valid_modalities
        if invalid:
            raise ValueError(
                f"Invalid modalities: {invalid}. "
                f"Valid options are: {valid_modalities}"
            )

        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.processed_path = config["processed_data"]

        # Load splits
        splits_path = os.path.join(self.processed_path, "splits.json")
        with open(splits_path, "r") as f:
            self.splits = json.load(f)

        # Validate city exists in splits
        if city not in self.splits:
            raise ValueError(
                f"City '{city}' not found in splits file. "
                f"Available cities: {list(self.splits.keys())}"
            )

        # Load metadata for this city
        metadata_path = os.path.join(self.processed_path, city, "metadata.gpkg")
        self.metadata = gpd.read_file(metadata_path, layer="Filtered_Buildings")

        # Build index of building indices for the requested split
        self.index = []
        city_splits = self.splits[city]
        for building_id, building_split in city_splits.items():
            if building_split == split:
                # Extract building index from "building_X" format
                idx = int(building_id.split("_")[1])
                self.index.append(idx)

        print(
            f"Loaded {len(self.index)} buildings from '{city}' for split '{split}' "
            f"with modalities: {self.modalities}"
        )

    def __len__(self):
        """Return the number of buildings in the dataset."""
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single building instance with candidates and their labels.

        Args:
            idx: Index into the dataset

        Returns:
            Dictionary containing building data with candidate entrance points
            and their binary labels
        """
        building_idx = self.index[idx]
        building = self.metadata.iloc[building_idx]

        # Build the result dictionary
        result = {
            "building_id": (self.city, building_idx),
            "geometry": building.geometry,
            "candidates": self._get_candidates(building),
            "labels": self._get_labels(building),
        }

        # Load modality-specific data
        if "streetview" in self.modalities:
            result["streetview_images"] = self._load_streetview(self.city, building)

        if "aerial" in self.modalities:
            result["aerial_image"] = self._load_aerial(self.city, building)

        if "gps" in self.modalities:
            result["gps_traces"] = self._load_gps(self.city, building)

        return result

    def _get_candidates(self, building) -> List[tuple]:
        """Extract candidate entrance points as (lon, lat) pairs.

        Args:
            building: Building row from metadata GeoDataFrame

        Returns:
            List of (longitude, latitude) tuples for candidate entrances
        """
        candidates_data = self._parse_candidates(building.get("candidates"))
        if not candidates_data or "all_points" not in candidates_data:
            return []

        coords = []
        for geom in candidates_data["all_points"]:
            # Candidate geometries are Point objects
            coords.append((geom.x, geom.y))

        return coords

    def _get_labels(self, building) -> List[int]:
        """Extract binary labels for candidate entrance points.

        Args:
            building: Building row from metadata GeoDataFrame

        Returns:
            List of binary labels (1=true entrance, 0=negative candidate)
        """
        candidates_data = self._parse_candidates(building.get("candidates"))
        if not candidates_data or "labels" not in candidates_data:
            return []

        return candidates_data["labels"]

    def _parse_candidates(self, value) -> Dict:
        """Parse serialized candidates payload from GeoPackage storage."""
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

        out = {
            "labels": list(payload.get("labels", [])),
            "all_points": [],
        }

        for item in payload.get("all_points", []):
            if isinstance(item, Point):
                out["all_points"].append(item)
                continue
            if not isinstance(item, str):
                continue
            try:
                geom = wkt.loads(item)
                if isinstance(geom, Point):
                    out["all_points"].append(geom)
            except Exception:
                continue

        return out

    def _load_streetview(self, city: str, building) -> List[Image.Image]:
        """Load streetview images for a building.

        Args:
            city: City name
            building: Building row from metadata GeoDataFrame

        Returns:
            List of PIL Images (may be empty)
        """
        images = []

        streetview_dir = building.get("streetview_image_dir", "")
        if not streetview_dir:
            return images

        full_dir = os.path.join(self.processed_path, city, streetview_dir)
        if not os.path.exists(full_dir):
            return images

        # Load all image files in the directory (matching preprocess step extensions)
        for fname in sorted(os.listdir(full_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(full_dir, fname)
                try:
                    img = Image.open(img_path).convert("RGB")
                    images.append(img)
                except Exception as e:
                    print(f"Warning: Failed to load {img_path}: {e}")

        return images

    def _load_aerial(self, city: str, building) -> Optional[np.ndarray]:
        """Load aerial image for a building.

        Args:
            city: City name
            building: Building row from metadata GeoDataFrame

        Returns:
            Numpy array of shape (C, H, W) or None if not available
        """
        aerial_path = building.get("aerial_image_path", "")
        if not aerial_path:
            return None

        full_path = os.path.join(self.processed_path, city, aerial_path)
        if not os.path.exists(full_path):
            return None

        try:
            with rasterio.open(full_path) as src:
                # Read all bands and return as (C, H, W)
                img = src.read()  # Shape: (bands, height, width)
                return img
        except Exception as e:
            print(f"Warning: Failed to load {full_path}: {e}")
            return None

    def _load_gps(self, city: str, building) -> Optional[pd.DataFrame]:
        """Load GPS trajectory traces for a building.

        Args:
            city: City name
            building: Building row from metadata GeoDataFrame

        Returns:
            DataFrame with GPS points or None if not available
        """
        gps_path = building.get("gps_trace_path", "")
        if not gps_path:
            return None

        full_path = os.path.join(self.processed_path, city, gps_path)
        if not os.path.exists(full_path):
            return None

        try:
            df = pd.read_csv(full_path)
            return df
        except Exception as e:
            print(f"Warning: Failed to load {full_path}: {e}")
            return None

    def get_building_metadata(self, idx: int) -> pd.Series:
        """Get the full metadata row for a building.

        Args:
            idx: Index into the dataset

        Returns:
            Pandas Series with all metadata fields
        """
        building_idx = self.index[idx]
        return self.metadata.iloc[building_idx]


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for batching buildings.

    Since buildings have variable numbers of candidates, we return lists
    rather than stacking into tensors.

    Args:
        batch: List of dictionaries from __getitem__

    Returns:
        Dictionary with batched data (mostly as lists)
    """
    result = {
        "building_id": [item["building_id"] for item in batch],
        "geometry": [item["geometry"] for item in batch],
        "candidates": [item["candidates"] for item in batch],
        "labels": [item["labels"] for item in batch],
    }

    # Check which modalities are present
    if "streetview_images" in batch[0]:
        result["streetview_images"] = [item["streetview_images"] for item in batch]

    if "aerial_image" in batch[0]:
        # Aerial images can be stacked if they have the same shape; keep None
        # entries to preserve alignment with other batched fields.
        aerial_imgs = [item["aerial_image"] for item in batch]
        result["aerial_image"] = aerial_imgs

    if "gps_traces" in batch[0]:
        result["gps_traces"] = [item["gps_traces"] for item in batch]

    return result


if __name__ == "__main__":
    # Example usage
    print("Loading train dataset for NYC with all modalities...")
    train_dataset = BuildingEntranceDataset(city="nyc", split="train")
    print(f"Train dataset size: {len(train_dataset)}")

    if len(train_dataset) > 0:
        print("\nExample building (binary classification for candidates):")
        sample = train_dataset[0]
        print(f"  Building ID: {sample['building_id']}")
        print(f"  Number of candidates: {len(sample['candidates'])}")
        print(f"  Labels: {sample['labels']}")
        print(f"  Positive samples (label=1): {sum(sample['labels'])}")
        print(
            f"  Negative samples (label=0): {len(sample['labels']) - sum(sample['labels'])}"
        )

        if "streetview_images" in sample:
            print(f"  Streetview images: {len(sample['streetview_images'])}")

        if "aerial_image" in sample and sample["aerial_image"] is not None:
            print(f"  Aerial image shape: {sample['aerial_image'].shape}")

        if "gps_traces" in sample and sample["gps_traces"] is not None:
            print(f"  GPS points: {len(sample['gps_traces'])}")

    print("\n" + "=" * 60)
    print("Loading val dataset for LA with only aerial and GPS...")
    val_dataset = BuildingEntranceDataset(
        city="la", split="val", modalities=["aerial", "gps"]
    )
    print(f"Val dataset size: {len(val_dataset)}")

    if len(val_dataset) > 0:
        sample = val_dataset[0]
        print(
            f"\nExample: building {sample['building_id']} has {len(sample['candidates'])} "
            f"candidates ({sum(sample['labels'])} positive, "
            f"{len(sample['labels']) - sum(sample['labels'])} negative)"
        )
