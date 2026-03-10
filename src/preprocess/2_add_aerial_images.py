"""Crop aerial (NAIP) image patches around each building footprint.

For each building in metadata.gpkg, finds the covering NAIP tile(s),
crops a buffered region around the building, and saves it as a GeoTIFF.
Adds an "aerial_image_path" column to the Filtered_Buildings layer.

Expected config keys in config.yaml:
- raw_data: path to raw data root containing a USGS/ directory
- processed_data: path to processed data root
- cities: list of city names
"""

import os
import zipfile

import geopandas as gpd
import numpy as np
import rasterio
import yaml
from pyproj import Transformer
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.windows import from_bounds as window_from_bounds
from shapely.geometry import box
from shapely.ops import transform as shapely_transform
from tqdm import tqdm

USGS_CITY_DIRS = {
    "nyc": "New York NAIP Images",
    "la": "Los Angeles NAIP Images",
    "chicago": "Chicago NAIP Images",
}

BUFFER_M = 50


def build_naip_index(city):
    """Build a spatial index of NAIP tile extents for a given city.

    Returns a list of dicts with keys: zip_path, tif_name, bounds (shapely box),
    crs (rasterio CRS).
    """
    usgs_dir = os.path.join(raw_path, "USGS", USGS_CITY_DIRS[city])
    tiles = []

    for fname in sorted(os.listdir(usgs_dir)):
        if not fname.upper().endswith(".ZIP"):
            continue

        zip_path = os.path.join(usgs_dir, fname)

        # Discover the .tif name inside the zip
        with zipfile.ZipFile(zip_path, "r") as zf:
            tif_names = [n for n in zf.namelist() if n.lower().endswith(".tif")]
            if not tif_names:
                continue
            tif_name = tif_names[0]

        uri = f"zip://{zip_path}!{tif_name}"
        with rasterio.open(uri) as src:
            b = src.bounds
            tiles.append(
                {
                    "zip_path": zip_path,
                    "tif_name": tif_name,
                    "bounds": box(b.left, b.bottom, b.right, b.top),
                    "crs": src.crs,
                }
            )

    print(f"  Indexed {len(tiles)} NAIP tiles for {city}")
    return tiles


def crop_building_aerial(building_geom, naip_tiles, buffer_m=BUFFER_M):
    """Crop an aerial image patch around a building.

    Args:
        building_geom: Building geometry in EPSG:4326.
        naip_tiles: List of tile dicts from build_naip_index.
        buffer_m: Buffer around the building footprint in meters.

    Returns:
        (data, profile) tuple or None if no covering tile found.
    """
    if not naip_tiles:
        return None

    # Use the CRS from the first tile (all tiles in a city share the same CRS)
    target_crs = naip_tiles[0]["crs"]

    # Reproject building geometry to the NAIP CRS for buffering in meters
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    projected_geom = shapely_transform(transformer.transform, building_geom)
    buffered = projected_geom.buffer(buffer_m)
    buffered_bounds = buffered.bounds  # (minx, miny, maxx, maxy)
    buffered_box = box(*buffered_bounds)

    # Find the tile with the largest overlap
    best_tile = None
    best_overlap = 0
    for tile in naip_tiles:
        overlap = tile["bounds"].intersection(buffered_box).area
        if overlap > best_overlap:
            best_overlap = overlap
            best_tile = tile

    if best_tile is None or best_overlap == 0:
        return None

    # Read the windowed raster data
    uri = f"zip://{best_tile['zip_path']}!{best_tile['tif_name']}"
    with rasterio.open(uri) as src:
        # Clamp bounds to the tile extent
        tile_b = src.bounds
        minx = max(buffered_bounds[0], tile_b.left)
        miny = max(buffered_bounds[1], tile_b.bottom)
        maxx = min(buffered_bounds[2], tile_b.right)
        maxy = min(buffered_bounds[3], tile_b.top)

        if minx >= maxx or miny >= maxy:
            return None

        window = window_from_bounds(minx, miny, maxx, maxy, src.transform)
        data = src.read(window=window)

        if data.size == 0:
            return None

        profile = src.profile.copy()
        profile.update(
            width=data.shape[2],
            height=data.shape[1],
            transform=transform_from_bounds(
                minx, miny, maxx, maxy, data.shape[2], data.shape[1]
            ),
        )

    return data, profile


def add_aerial_images_one_city(city):
    """Process one city: crop aerial patches and update metadata."""
    print(f"Processing city: {city}")
    metadata_path = os.path.join(processed_path, city, "metadata.gpkg")
    gdf = gpd.read_file(metadata_path, layer="Filtered_Buildings")

    naip_tiles = build_naip_index(city)

    aerial_dir = os.path.join(processed_path, city, "aerial")
    os.makedirs(aerial_dir, exist_ok=True)

    paths = []
    for idx, row in tqdm(
        enumerate(gdf.itertuples()), total=len(gdf), desc=f"  Cropping {city}"
    ):
        result = crop_building_aerial(row.geometry, naip_tiles)
        if result is None:
            paths.append("")
            continue

        data, profile = result
        out_file = f"building_{idx}.tif"
        out_path = os.path.join(aerial_dir, out_file)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)
        paths.append(os.path.join("aerial", out_file))

    gdf["aerial_image_path"] = paths

    n_found = sum(1 for p in paths if p)
    print(f"  {n_found}/{len(gdf)} buildings matched to aerial tiles")

    gdf.to_file(metadata_path, layer="Filtered_Buildings", driver="GPKG")
    print(f"  Updated metadata: {metadata_path}\n")


def add_aerial_images(cities):
    for city in cities:
        add_aerial_images_one_city(city)


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    raw_path = config["raw_data"]
    processed_path = config["processed_data"]
    cities = config["cities"]

    add_aerial_images(cities)
