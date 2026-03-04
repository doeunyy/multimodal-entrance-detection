"""
Configuration for the building entrance tagging pipeline.

Schema: single layer 'Buildings_With_Entrances' per GeoPackage with columns:
  osm_way_id, entrance_node_ids, entrance_geometries (list of WKT strings),
  match_tiers (list of ints), n_entrances, building, name, addr:street,
  addr:housenumber, addr:city, addr:postcode, geometry (Polygon footprint)
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    gpkg_paths: List[str] = field(default_factory=lambda: [
        "C:/Users/Andrew/Desktop/multimodal-entrance-detection/data/processed/chicago/metadata.gpkg",
        "C:/Users/Andrew/Desktop/multimodal-entrance-detection/data/processed/la/metadata.gpkg",
        "C:/Users/Andrew/Desktop/multimodal-entrance-detection/data/processed/nyc/metadata.gpkg",
    ])
    layer_name: str = "Buildings_With_Entrances"
    output_dir: str = "outputs"


@dataclass
class PreprocessConfig:
    segment_interval: float = 3.0           # metres; footprint split interval
    physical_distance_threshold: float = 24.0  # PT in paper (metres)
    feature_distance_threshold: float = 0.04   # FT in paper (normalised)


@dataclass
class ModelConfig:
    n_folds: int = 5
    random_state: int = 42
    # Weighted Random Forest
    wrf_n_trees: int = 80
    wrf_max_depth: int = 12
    wrf_class_weight_ratio: int = 160
    # Balanced Random Forest
    brf_n_trees: int = 140
    brf_max_depth: int = 14
    # SmoteBoost
    smote_n_estimators: int = 90
    smote_k_neighbors: int = 4
    # Baseline Random Forest
    rf_n_trees: int = 110
    rf_max_depth: int = 14


CITY_CRS = {
    "chicago": "EPSG:32616",
    "la":      "EPSG:32611",
    "nyc":     "EPSG:32618",
}