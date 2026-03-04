"""
Data loading: parse GeoPackage files and extract building + entrance data.

Each GeoPackage has a single layer 'Buildings_With_Entrances' where
entrance_geometries is a string-encoded Python list of WKT point strings
and match_tiers is a string-encoded list of integers.
"""
import ast
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import MultiPolygon, Point, Polygon

from config import CITY_CRS, DataConfig

log = logging.getLogger(__name__)


def _parse_str_list(s: str) -> list:
    """Safely parse a string like \"['POINT (...)', 'POINT (...)']\" into a list."""
    if pd.isna(s) or s is None:
        return []
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []


def _pick_main_entrance_wkt(entrance_geoms_str: str, tiers_str: str) -> Optional[str]:
    """
    Choose the main entrance from a building's entrance list.

    Strategy: pick the first tier-1 entrance; if none, pick the first entrance.
    """
    geoms = _parse_str_list(entrance_geoms_str)
    tiers = _parse_str_list(tiers_str)
    if not geoms:
        return None
    # Pair them up
    if len(tiers) == len(geoms):
        for g, t in zip(geoms, tiers):
            if t == 1:
                return g
    # Fallback to first entrance
    return geoms[0]


def _detect_city(gpkg_path: str) -> str:
    """Infer city name from the file path for CRS selection."""
    # Check the full path (including parent directories), not just filename
    full_path = str(Path(gpkg_path).resolve()).lower().replace("\\", "/")
    for city in CITY_CRS:
        if city in full_path:
            return city
    log.warning("Could not detect city from path '%s', defaulting to chicago", gpkg_path)
    return "chicago"


def load_city(
    gpkg_path: str,
    cfg: DataConfig,
) -> Tuple[gpd.GeoDataFrame, str]:
    """
    Load one city's GeoPackage and return a projected GeoDataFrame with columns:
      building_id, geometry (Polygon, projected), entrance_pt (Point, projected),
      building_type, name, addr_street, city

    Also returns the projected CRS string.
    """
    city = _detect_city(gpkg_path)
    crs = CITY_CRS.get(city, "EPSG:32616")

    raw = gpd.read_file(gpkg_path, layer=cfg.layer_name)
    log.info("  %s: %d buildings loaded (CRS %s -> %s)", city, len(raw), raw.crs, crs)

    rows = []
    for idx, r in raw.iterrows():
        fp = r.geometry
        if fp is None or fp.is_empty:
            continue
        if isinstance(fp, MultiPolygon):
            fp = max(fp.geoms, key=lambda g: g.area)

        main_wkt = _pick_main_entrance_wkt(
            r.get("entrance_geometries", ""),
            r.get("match_tiers", ""),
        )
        if main_wkt is None:
            continue

        try:
            entrance_pt = wkt.loads(main_wkt)
        except Exception:
            continue

        rows.append({
            "osm_way_id": r.get("osm_way_id"),
            "geometry": fp,
            "entrance_pt": entrance_pt,
            "building_type": r.get("building", ""),
            "name": r.get("name", ""),
            "addr_street": r.get("addr:street", ""),
            "city": city,
        })

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=raw.crs)
    gdf = gdf.to_crs(crs)

    # Project entrance points too
    ent_gdf = gpd.GeoDataFrame(
        geometry=[r["entrance_pt"] for r in rows], crs=raw.crs
    ).to_crs(crs)
    gdf["entrance_pt"] = ent_gdf.geometry.values

    # Assign unique building IDs
    gdf["building_id"] = np.arange(len(gdf))

    log.info("  %s: %d buildings with valid main entrance", city, len(gdf))
    return gdf, crs


def load_all_cities(cfg: DataConfig) -> dict:
    """
    Load all cities.

    Returns dict: city_name -> GeoDataFrame (each in its own projected CRS).
    """
    city_gdfs = {}
    bid_offset = 0

    for gpkg_path in cfg.gpkg_paths:
        log.info("Loading %s", gpkg_path)
        gdf, crs = load_city(gpkg_path, cfg)
        gdf["building_id"] += bid_offset
        bid_offset += len(gdf)
        city = gdf["city"].iloc[0]
        city_gdfs[city] = gdf

    total = sum(len(g) for g in city_gdfs.values())
    log.info("Total buildings across all cities: %d", total)
    return city_gdfs