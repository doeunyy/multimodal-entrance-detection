"""
Mapillary Step 3

Assuming streetview images are already downloaded into:

  data/raw/Mapillary/Mapillary/<city>/streetview/building_{idx}/

where {idx} matches the row index of the building in metadata.gpkg.

This script:
1) Copies those images into data/processed/<city>/streetview/building_{idx}/
2) Updates metadata.gpkg (written to layer "Filtered_Buildings") with:
- streetview_image_dir (e.g., "streetview/building_0")
- n_streetview_images (int)
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Optional

import geopandas as gpd
import yaml
from tqdm import tqdm


def _list_gpkg_layers(gpkg_path: str) -> list[str]:
    """Return all layer names inside a gpkg file."""
    import fiona

    return list(fiona.listlayers(gpkg_path))


def _choose_building_layer(gpkg_path: str) -> str:
    """
    Different preprocessing steps may name the layer slightly differently.
    We prefer 'Filtered_Buildings', but fall back safely if needed.
    """
    layers = _list_gpkg_layers(gpkg_path)
    if "Filtered_Buildings" in layers:
        return "Filtered_Buildings"
    if "Buildings_With_Entrances" in layers:
        return "Buildings_With_Entrances"
    if not layers:
        raise ValueError(f"No layers found in {gpkg_path}")
    print(f"  [WARN] Unexpected layers {layers}. Using first layer: {layers[0]}")
    return layers[0]


def _atomic_write_gpkg(gdf: gpd.GeoDataFrame, out_path: str, layer: str) -> None:
    """
    Write gpkg safely:
    write to a temp file first, then replace the original.
    This avoids file-lock issues on Windows.
    """
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(suffix=".gpkg", dir=out_dir)
    os.close(fd)
    try:
        gdf.to_file(tmp_path, layer=layer, driver="GPKG")
        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _count_images(folder: str) -> int:
    """Count image files in a folder (jpg/jpeg/png)."""
    if not os.path.isdir(folder):
        return 0
    exts = (".jpg", ".jpeg", ".png")
    return sum(
        1
        for fn in os.listdir(folder)
        if fn.lower().endswith(exts) and os.path.isfile(os.path.join(folder, fn))
    )


def _find_raw_streetview_root(raw_path: str, city: str) -> str:
    candidates = [
        city,
        city.lower(),
        city.upper(),
        city.title(),
    ]
    for c in candidates:
        p = os.path.join(raw_path, "Mapillary", "Mapillary", c, "streetview")
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(
        "Could not find raw streetview folder. Tried:\n"
        + "\n".join(
            [
                os.path.join(raw_path, "Mapillary", "Mapillary", c, "streetview")
                for c in candidates
            ]
        )
    )


def link_streetview_images_one_city(
    raw_path: str, processed_path: str, city: str
) -> None:
    print(f"\nProcessing city: {city}")

    metadata_path = os.path.join(processed_path, city, "metadata.gpkg")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f"metadata.gpkg not found for {city}: {metadata_path}\n"
            f"Before running Mapillary, ensure OSM/USGS produced it under data/processed/{city}/metadata.gpkg "
            f"(or copy the base metadata there)."
        )

    # Load the metadata layer
    in_layer = _choose_building_layer(metadata_path)
    gdf = gpd.read_file(metadata_path, layer=in_layer)

    # Ensure idx is exactly row index
    gdf = gdf.reset_index(drop=True)

    # Find raw streetview folder
    raw_streetview_root = _find_raw_streetview_root(raw_path, city)
    print(f"  Using raw streetview root: {raw_streetview_root}")

    streetview_dirs: list[str] = []
    n_images_list: list[int] = []

    for idx in tqdm(range(len(gdf)), desc=f"  Linking {city}", total=len(gdf)):
        src_dir = os.path.join(raw_streetview_root, f"building_{idx}")
        n_src = _count_images(src_dir)

        if n_src == 0:
            streetview_dirs.append("")
            n_images_list.append(0)
            continue

        out_dir = os.path.join(processed_path, city, "streetview", f"building_{idx}")
        os.makedirs(out_dir, exist_ok=True)

        copied = 0
        for fn in os.listdir(src_dir):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            src = os.path.join(src_dir, fn)
            dst = os.path.join(out_dir, fn)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                copied += 1

        if copied > 0:
            streetview_dirs.append(os.path.join("streetview", f"building_{idx}"))
        else:
            streetview_dirs.append("")
        n_images_list.append(copied)

    gdf["streetview_image_dir"] = streetview_dirs
    gdf["n_streetview_images"] = n_images_list

    n_with_images = sum(1 for n in n_images_list if n > 0)
    total_images = sum(n_images_list)
    print(f"  {n_with_images}/{len(gdf)} buildings have streetview images")
    print(f"  {total_images} total images copied")

    out_layer = "Filtered_Buildings"
    _atomic_write_gpkg(gdf, metadata_path, layer=out_layer)
    print(f"  Updated metadata: {metadata_path} (layer: {out_layer})")


def main() -> None:
    # Load config.yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    raw_path = config["raw_data"]
    processed_path = config["processed_data"]
    cities = config["cities"]

    for city in cities:
        link_streetview_images_one_city(raw_path, processed_path, city)


if __name__ == "__main__":
    main()
