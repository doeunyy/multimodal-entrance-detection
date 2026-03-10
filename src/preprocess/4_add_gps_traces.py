"""Attach GPS trajectory statistics and nearby trace segments to each building.

For each building in metadata.gpkg, reads WorldTrace trajectory-per-building
statistics (P50), joins by osm_id, and extracts nearby trajectory segments.
Adds trajectory count columns and a path to extracted GPS data.

Expected config keys in config.yaml:
- raw_data: path to raw data root containing a WorldTrace/ directory
- processed_data: path to processed data root
- cities: list of city names
"""

import os
from math import cos, radians

import geopandas as gpd
import pandas as pd
import yaml
from tqdm import tqdm

BUFFER_M = 50


def normalize_osm_id(value):
    """Normalize OSM ids to stable string keys for cross-source joins."""
    if pd.isna(value):
        return ""

    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return ""

    # Some exporters serialize integer ids as floats (e.g. 12345.0).
    if s.endswith(".0") and s[:-2].isdigit():
        return s[:-2]
    return s


def safe_int(value, default=0):
    """Convert numeric values to int while tolerating NaN/None/string noise."""
    if pd.isna(value):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def meters_to_degree_offsets(lat_deg, meters):
    """Convert meter offsets to approx lon/lat degree offsets at latitude."""
    lat_offset = meters / 111320.0
    cos_lat = max(cos(radians(lat_deg)), 1e-6)
    lon_offset = meters / (111320.0 * cos_lat)
    return lon_offset, lat_offset


def load_trajectory_stats(city):
    """Load pre-computed trajectory statistics from the P50 CSV.

    Normalizes id columns to stable string keys to match metadata.gpkg.
    """
    paths = [
        os.path.join(raw_path, "WorldTrace", city, "Trajectory per building P50.csv"),
        os.path.join(raw_path, "WorldTrace", city, "traj_per_building_p50.csv"),
        os.path.join(raw_path, "WorldTrace", city, "traj_per_building.csv"),
    ]

    csv_path = next((p for p in paths if os.path.isfile(p)), None)
    if csv_path is None:
        raise FileNotFoundError(f"No trajectory stats file found for {city}")

    df = pd.read_csv(csv_path)

    id_col = next(
        (c for c in ["osm_id", "osm_way_id", "building_way_id"] if c in df.columns),
        None,
    )
    if id_col is None:
        raise ValueError(
            "No usable building id column found in "
            f"{csv_path}. Columns: {df.columns.tolist()}"
        )

    if id_col != "osm_id":
        df = df.rename(columns={id_col: "osm_id"})

    df["osm_id"] = df["osm_id"].map(normalize_osm_id)
    df = df[df["osm_id"] != ""].copy()
    print(
        f"  Loaded trajectory stats: {len(df)} buildings from {os.path.basename(csv_path)}"
    )

    return df


def load_trajectory_index(city):
    """Load the trajectory index with bounding boxes.

    Fixes Windows-style path separators in traj_relpath.
    """
    csv_path = os.path.join(raw_path, "WorldTrace", city, "index.csv")
    df = pd.read_csv(csv_path)
    df["traj_relpath"] = df["traj_relpath"].str.replace("\\", "/", regex=False)
    print(f"  Loaded trajectory index: {len(df)} trajectories")
    return df


def find_nearby_trajectories(building_geom, traj_index_df, buffer_m=BUFFER_M):
    """Find trajectories whose bounding boxes intersect the building's buffered bbox.

    Uses an approximate lon/lat bbox expanded by buffer_m for fast pre-filtering.

    Returns list of traj_relpath values.
    """
    bx = building_geom.bounds  # (minx, miny, maxx, maxy)
    lon_offset, lat_offset = meters_to_degree_offsets(
        building_geom.centroid.y, buffer_m
    )
    buf_minx = bx[0] - lon_offset
    buf_miny = bx[1] - lat_offset
    buf_maxx = bx[2] + lon_offset
    buf_maxy = bx[3] + lat_offset

    mask = (
        (traj_index_df["bbox_minx"] <= buf_maxx)
        & (traj_index_df["bbox_maxx"] >= buf_minx)
        & (traj_index_df["bbox_miny"] <= buf_maxy)
        & (traj_index_df["bbox_maxy"] >= buf_miny)
    )
    return traj_index_df.loc[mask, "traj_relpath"].tolist()


def extract_trajectory_segments(building_geom, traj_relpaths, city, buffer_m=BUFFER_M):
    """Read matching trajectory CSVs and filter points near the building.

    Uses true metric distance to the building footprint (buffer_m in meters).

    Returns a DataFrame of nearby trajectory points with an added traj_id column.
    """
    centroid = building_geom.centroid
    cx, cy = centroid.x, centroid.y
    lon_offset, lat_offset = meters_to_degree_offsets(cy, buffer_m)

    # Project once per building for robust meter-distance filtering.
    building_gs = gpd.GeoSeries([building_geom], crs="EPSG:4326")
    utm_crs = building_gs.estimate_utm_crs()
    building_buffered = building_gs.to_crs(utm_crs).buffer(buffer_m).iloc[0]

    segments = []
    traj_dir = os.path.join(raw_path, "WorldTrace", city, "trajectories")

    for relpath in traj_relpaths:
        traj_path = os.path.join(traj_dir, relpath)
        if not os.path.isfile(traj_path):
            continue

        df = pd.read_csv(traj_path)
        if "latitude" not in df.columns or "longitude" not in df.columns:
            continue

        # Fast coarse filter in lon/lat, then exact metric filter on building buffer.
        coarse_mask = (
            (df["longitude"] >= cx - lon_offset)
            & (df["longitude"] <= cx + lon_offset)
            & (df["latitude"] >= cy - lat_offset)
            & (df["latitude"] <= cy + lat_offset)
        )
        coarse = df.loc[coarse_mask].copy()
        if coarse.empty:
            continue

        points = gpd.GeoSeries(
            gpd.points_from_xy(coarse["longitude"], coarse["latitude"]),
            crs="EPSG:4326",
        ).to_crs(utm_crs)
        near_mask = points.within(building_buffered)
        nearby = coarse.loc[near_mask.values]
        if not nearby.empty:
            traj_id = os.path.splitext(os.path.basename(relpath))[0]
            nearby = nearby.copy()
            nearby.insert(0, "traj_id", traj_id)
            segments.append(nearby)

    if segments:
        return pd.concat(segments, ignore_index=True)
    return pd.DataFrame()


def add_gps_traces_one_city(city):
    """Process one city: join trajectory stats and extract nearby segments."""
    print(f"Processing city: {city}")
    metadata_path = os.path.join(processed_path, city, "metadata.gpkg")
    layer_paths = ["Filtered_Buildings", "Buildings_With_Entrances"]
    gdf = None
    loaded_layer = None
    last_err = None

    for layer_name in layer_paths:
        try:
            gdf = gpd.read_file(metadata_path, layer=layer_name)
            loaded_layer = layer_name
            print(f"  Loaded metadata layer: {layer_name}")
            break
        except Exception as e:
            last_err = e

    if gdf is None:
        raise RuntimeError(
            f"Could not read a valid building layer from {metadata_path}. "
            f"Tried {layer_paths}. Last error: {last_err}"
        )

    # Join pre-computed trajectory stats
    stats_df = load_trajectory_stats(city)
    if not stats_df["osm_id"].is_unique:
        dup_count = stats_df["osm_id"].duplicated().sum()
        print(
            f"  Found {dup_count} duplicate osm_id rows in trajectory stats; collapsing with max()."
        )
        stats_df = stats_df.groupby("osm_id", as_index=False)[
            ["n_traj_within_50m", "n_traj_inside"]
        ].max()

    stats_lookup = stats_df.set_index("osm_id")[
        ["n_traj_within_50m", "n_traj_inside"]
    ].to_dict("index")

    traj_index_df = load_trajectory_index(city)

    gps_dir = os.path.join(processed_path, city, "gps")
    os.makedirs(gps_dir, exist_ok=True)

    n_within = []
    n_inside = []
    gps_paths = []

    id_col = None
    stats_ids = set(stats_df["osm_id"])
    candidate_cols = [c for c in ["osm_way_id", "osm_id"] if c in gdf.columns]
    candidate_overlaps = []

    for candidate in candidate_cols:
        candidate_ids = {
            normalize_osm_id(v) for v in gdf[candidate].tolist() if normalize_osm_id(v)
        }
        overlap = len(candidate_ids & stats_ids)
        candidate_overlaps.append((candidate, overlap))

    if candidate_overlaps:
        id_col, overlap = max(candidate_overlaps, key=lambda x: x[1])
        overlap_str = ", ".join(f"{name}={count}" for name, count in candidate_overlaps)
        print(f"  Metadata id overlap with trajectory stats: {overlap_str}")
        if overlap == 0:
            raise ValueError(
                "Could not find overlapping IDs between metadata and trajectory stats. "
                f"Tried columns: {[name for name, _ in candidate_overlaps]}"
            )

    if id_col is None:
        raise ValueError(
            f"No usable OSM id column found in metadata. Columns: {gdf.columns.tolist()}"
        )

    print(f"  Using metadata id column: {id_col}")

    for idx, row in tqdm(
        enumerate(gdf.itertuples()), total=len(gdf), desc=f"  Extracting {city}"
    ):
        osm_id = normalize_osm_id(getattr(row, id_col))
        stats = stats_lookup.get(osm_id, {})
        n_w = safe_int(stats.get("n_traj_within_50m", 0))
        n_i = safe_int(stats.get("n_traj_inside", 0))
        n_within.append(n_w)
        n_inside.append(n_i)

        # Only extract segments if there are nearby trajectories
        if n_w > 0:
            nearby_relpaths = find_nearby_trajectories(row.geometry, traj_index_df)
            if nearby_relpaths:
                segments_df = extract_trajectory_segments(
                    row.geometry, nearby_relpaths, city
                )
                if not segments_df.empty:
                    out_file = f"building_{idx}.csv"
                    segments_df.to_csv(os.path.join(gps_dir, out_file), index=False)
                    gps_paths.append(os.path.join("gps", out_file))
                    continue

        gps_paths.append("")

    gdf["n_traj_within_50m"] = n_within
    gdf["n_traj_inside"] = n_inside
    gdf["gps_trace_path"] = gps_paths

    n_with_traces = sum(1 for p in gps_paths if p)
    n_with_stats = sum(1 for n in n_within if n > 0)
    print(f"  {n_with_stats}/{len(gdf)} buildings have trajectory stats")
    print(f"  {n_with_traces}/{len(gdf)} buildings have extracted GPS segments")

    gdf.to_file(metadata_path, layer=loaded_layer, driver="GPKG")
    print(f"  Updated metadata: {metadata_path}\n")


def add_gps_traces(cities):
    for city in cities:
        add_gps_traces_one_city(city)


if __name__ == "__main__":
    # :3
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    raw_path = config["raw_data"]
    processed_path = config["processed_data"]
    cities = config["cities"]

    add_gps_traces(cities)
