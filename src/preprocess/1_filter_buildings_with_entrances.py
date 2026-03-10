"""OSM Building-Entrance Matching Pipeline.

For each entrance node, find the corresponding building using tiered matching:
  Tier 1 — Topological: entrance node is a vertex of the building way
  Tier 2 — Address: match on addr:street + addr:housenumber tags
  Tier 3 — Spatial: entrance within 0.5m of building boundary (fallback)

Then aggregate results so each building has a list of its entrance geometries.
"""

import os
from collections import defaultdict

import geopandas as gpd
import osmium
import pandas as pd
import yaml
from shapely.geometry import Point, Polygon

# =============================================================================
# Step 1: Parse PBF files
# =============================================================================


class OSMBuildingEntranceHandler(osmium.SimpleHandler):
    """Single-pass PBF handler that collects buildings, entrances, and their
    topological relationships (which entrance nodes are vertices of which
    building ways)."""

    def __init__(self):
        super().__init__()
        self.node_coords: dict[int, tuple[float, float]] = {}
        self.entrance_nodes: dict[int, dict] = {}
        self.building_ways: dict[int, dict] = {}
        self.entrance_to_ways: dict[int, list[int]] = defaultdict(list)

    def node(self, n):
        self.node_coords[n.id] = (n.location.lon, n.location.lat)
        if "entrance" in n.tags:
            self.entrance_nodes[n.id] = {
                "tags": dict(n.tags),
                "lon": n.location.lon,
                "lat": n.location.lat,
            }

    def way(self, w):
        if "building" not in w.tags:
            return
        node_ids = [n.ref for n in w.nodes]
        self.building_ways[w.id] = {
            "tags": dict(w.tags),
            "node_ids": node_ids,
        }
        for nid in node_ids:
            if nid in self.entrance_nodes:
                self.entrance_to_ways[nid].append(w.id)


def parse_pbf(pbf_path: str) -> OSMBuildingEntranceHandler:
    """Parse a PBF file and return the handler with all collected data."""
    handler = OSMBuildingEntranceHandler()
    handler.apply_file(pbf_path, locations=True, idx="flex_mem")
    print(
        f"  Parsed: {len(handler.building_ways):,} buildings, "
        f"{len(handler.entrance_nodes):,} entrances"
    )
    return handler


def handler_to_gdfs(handler: OSMBuildingEntranceHandler):
    """Convert parsed handler data into GeoDataFrames."""
    # Entrances
    ent_records = []
    for nid, data in handler.entrance_nodes.items():
        ent_records.append(
            {
                "osm_node_id": nid,
                "geometry": Point(data["lon"], data["lat"]),
                **data["tags"],
            }
        )
    entrance_gdf = (
        gpd.GeoDataFrame(ent_records, crs="EPSG:4326")
        if ent_records
        else gpd.GeoDataFrame(
            columns=["osm_node_id", "geometry"], geometry="geometry", crs="EPSG:4326"
        )
    )

    # Buildings
    bldg_records = []
    for wid, data in handler.building_ways.items():
        coords = [
            handler.node_coords[nid]
            for nid in data["node_ids"]
            if nid in handler.node_coords
        ]
        if len(coords) < 4:
            continue
        bldg_records.append(
            {
                "osm_way_id": wid,
                "geometry": Polygon(coords),
                **data["tags"],
            }
        )
    building_gdf = (
        gpd.GeoDataFrame(bldg_records, crs="EPSG:4326")
        if bldg_records
        else gpd.GeoDataFrame(
            columns=["osm_way_id", "geometry"], geometry="geometry", crs="EPSG:4326"
        )
    )

    return building_gdf, entrance_gdf


# =============================================================================
# Step 2: Tiered matching
# =============================================================================


def tier1_topological(handler: OSMBuildingEntranceHandler) -> pd.DataFrame:
    """Tier 1: entrance node is literally a vertex of the building way."""
    records = []
    for nid, way_ids in handler.entrance_to_ways.items():
        for wid in way_ids:
            records.append(
                {
                    "entrance_node_id": nid,
                    "building_way_id": wid,
                    "match_tier": 1,
                }
            )
    df = pd.DataFrame(records)
    n_matches = len(df)
    n_unique = df["entrance_node_id"].nunique() if n_matches else 0
    print(
        f"  Tier 1 (topological): {n_matches:,} matches "
        f"({n_unique:,} unique entrances)"
    )
    return df


def tier2_address(entrance_gdf, building_gdf, already_matched_ids: set) -> pd.DataFrame:
    """Tier 2: match on addr:street + addr:housenumber for unmatched entrances."""
    empty = pd.DataFrame(columns=["entrance_node_id", "building_way_id", "match_tier"])
    addr_cols = ["addr:street", "addr:housenumber"]

    if not all(c in entrance_gdf.columns for c in addr_cols):
        print("  Tier 2 (address): skipped — entrances lack address tags")
        return empty
    if not all(c in building_gdf.columns for c in addr_cols):
        print("  Tier 2 (address): skipped — buildings lack address tags")
        return empty

    remaining = entrance_gdf[~entrance_gdf["osm_node_id"].isin(already_matched_ids)]
    has_addr = remaining.dropna(subset=addr_cols, how="any")

    if has_addr.empty:
        print("  Tier 2 (address): 0 matches (no remaining entrances with addresses)")
        return empty

    bldg_with_addr = building_gdf[["osm_way_id"] + addr_cols].dropna(
        subset=addr_cols, how="any"
    )

    merged = has_addr[["osm_node_id"] + addr_cols].merge(
        bldg_with_addr,
        on=addr_cols,
        how="inner",
    )
    records = merged.rename(
        columns={
            "osm_node_id": "entrance_node_id",
            "osm_way_id": "building_way_id",
        }
    )[["entrance_node_id", "building_way_id"]].copy()
    records["match_tier"] = 2

    n_unique = records["entrance_node_id"].nunique() if len(records) else 0
    print(
        f"  Tier 2 (address): {len(records):,} matches "
        f"({n_unique:,} unique entrances)"
    )
    return records


def tier3_spatial(
    entrance_gdf, building_gdf, already_matched_ids: set, tol_m: float = 0.5
) -> pd.DataFrame:
    """Tier 3: entrance within tol_m of building polygon (buffered)."""
    empty = pd.DataFrame(columns=["entrance_node_id", "building_way_id", "match_tier"])

    remaining = entrance_gdf[~entrance_gdf["osm_node_id"].isin(already_matched_ids)]
    if remaining.empty:
        print("  Tier 3 (spatial): no remaining entrances")
        return empty

    utm_crs = building_gdf.estimate_utm_crs()
    b_proj = building_gdf[["osm_way_id", "geometry"]].to_crs(utm_crs).copy()
    e_proj = remaining[["osm_node_id", "geometry"]].to_crs(utm_crs)

    b_proj["geometry"] = b_proj.geometry.buffer(tol_m)

    joined = gpd.sjoin(e_proj, b_proj, predicate="within", how="inner")

    records = joined.rename(
        columns={
            "osm_node_id": "entrance_node_id",
            "osm_way_id": "building_way_id",
        }
    )[["entrance_node_id", "building_way_id"]].copy()
    records["match_tier"] = 3

    n_unique = records["entrance_node_id"].nunique() if len(records) else 0
    print(
        f"  Tier 3 (spatial): {len(records):,} matches "
        f"({n_unique:,} unique entrances)"
    )
    return records


def match_entrances_to_buildings(handler, building_gdf, entrance_gdf) -> pd.DataFrame:
    """Run all tiers sequentially and combine results."""
    # Tier 1
    t1 = tier1_topological(handler)
    matched_ids = set(t1["entrance_node_id"]) if len(t1) else set()

    # Tier 2
    t2 = tier2_address(entrance_gdf, building_gdf, matched_ids)
    matched_ids |= set(t2["entrance_node_id"]) if len(t2) else set()

    # Tier 3
    t3 = tier3_spatial(entrance_gdf, building_gdf, matched_ids)
    matched_ids |= set(t3["entrance_node_id"]) if len(t3) else set()

    all_matches = pd.concat([t1, t2, t3], ignore_index=True)

    total = len(entrance_gdf)
    unmatched = total - len(matched_ids)
    print(
        f"\n  Summary: {len(matched_ids):,}/{total:,} entrances matched, "
        f"{unmatched:,} unmatched"
    )

    return all_matches


def filter_by_distance(
    matches_df, building_gdf, entrance_gdf, max_dist_m: float = 500.0
) -> pd.DataFrame:
    """Drop matches where the entrance is more than max_dist_m from the
    building centroid. Catches misplaced OSM nodes and corrupted geometries."""
    if matches_df.empty:
        return matches_df

    before = len(matches_df)

    # Build lookup dicts
    eid_to_geom = entrance_gdf.set_index("osm_node_id")["geometry"].to_dict()
    bid_to_geom = building_gdf.set_index("osm_way_id")["geometry"].to_dict()

    # Project to meters
    utm_crs = building_gdf.estimate_utm_crs()

    # Vectorized approach: project all at once
    entrance_pts = matches_df["entrance_node_id"].map(eid_to_geom)
    building_centroids = matches_df["building_way_id"].map(
        {wid: geom.centroid for wid, geom in bid_to_geom.items()}
    )

    e_series = gpd.GeoSeries(entrance_pts.values, crs="EPSG:4326").to_crs(utm_crs)
    b_series = gpd.GeoSeries(building_centroids.values, crs="EPSG:4326").to_crs(utm_crs)

    distances = e_series.distance(b_series)
    mask = distances <= max_dist_m

    filtered = matches_df[mask.values].copy()
    removed = before - len(filtered)

    if removed:
        print(
            f"\n  Distance filter (max {max_dist_m:.0f}m): removed {removed:,} matches"
        )
        # Show worst offenders
        worst = distances[~mask.values].sort_values(ascending=False).head(5)
        for idx in worst.index:
            row = matches_df.iloc[idx]
            print(
                f"    - entrance {row['entrance_node_id']} → building {row['building_way_id']}: "
                f"{distances.iloc[idx]:.0f}m (tier {row['match_tier']})"
            )
    else:
        print(
            f"\n  Distance filter (max {max_dist_m:.0f}m): all matches within threshold"
        )

    return filtered


# =============================================================================
# Step 3: Aggregate and save
# =============================================================================


def aggregate_and_save(matches_df, building_gdf, entrance_gdf, output_path):
    """Attach entrance geometries to buildings and write GeoPackage."""
    if matches_df.empty:
        print("  No matches to save.")
        return

    eid_to_geom = entrance_gdf.set_index("osm_node_id")["geometry"].to_dict()
    matches_df = matches_df.copy()
    matches_df["entrance_geometry"] = matches_df["entrance_node_id"].map(eid_to_geom)

    grouped = matches_df.groupby("building_way_id").agg(
        entrance_node_ids=("entrance_node_id", list),
        entrance_geometries=("entrance_geometry", list),
        match_tiers=("match_tier", list),
        n_entrances=("entrance_node_id", "count"),
    )

    # Convert lists to strings for GeoPackage compatibility
    grouped["entrance_node_ids"] = grouped["entrance_node_ids"].apply(str)
    grouped["entrance_geometries"] = grouped["entrance_geometries"].apply(
        lambda pts: str([p.wkt for p in pts])
    )
    grouped["match_tiers"] = grouped["match_tiers"].apply(str)

    result = building_gdf.merge(
        grouped, left_on="osm_way_id", right_index=True, how="inner"
    )

    print(f"  Output: {len(result):,} buildings with entrances")

    # Keep only the columns we need — raw OSM tags can have types that
    # GeoPackage/pyogrio cannot serialize (e.g. mixed-type 'notes' fields).
    keep_cols = [
        "osm_way_id",
        "geometry",
        "entrance_node_ids",
        "entrance_geometries",
        "match_tiers",
        "n_entrances",
    ]
    # Optionally preserve common useful OSM tags if they exist
    for col in [
        "building",
        "name",
        "addr:street",
        "addr:housenumber",
        "addr:city",
        "addr:postcode",
    ]:
        if col in result.columns:
            keep_cols.append(col)

    result = result[[c for c in keep_cols if c in result.columns]]

    # Cast osm_way_id to string so GeoPackage doesn't absorb it as internal fid
    result["osm_way_id"] = result["osm_way_id"].astype(str)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_layer = "Filtered_Buildings"
    result.to_file(output_path, layer=output_layer, driver="GPKG")
    print(f"  Saved to: {output_path} (layer: {output_layer})")


# =============================================================================
# Main
# =============================================================================


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    raw_path = config["raw_data"]
    processed_path = config["processed_data"]
    cities = config["cities"]
    max_dist_m = config.get("max_dist_m", 500.0)

    for city_name in cities:
        pbf_file = os.path.join(raw_path, "OSM", f"{city_name}.osm.pbf")
        print(f"\n{'=' * 60}")
        print(f"Processing: {city_name}")
        print(f"{'=' * 60}")

        if not os.path.exists(pbf_file):
            print(f"  ERROR: {pbf_file} not found — skipping.")
            continue

        # Parse
        handler = parse_pbf(pbf_file)
        building_gdf, entrance_gdf = handler_to_gdfs(handler)

        if entrance_gdf.empty:
            print("  No entrances found — skipping.")
            continue

        # Match
        matches = match_entrances_to_buildings(handler, building_gdf, entrance_gdf)

        # Filter out matches that are too far from their building
        matches = filter_by_distance(
            matches, building_gdf, entrance_gdf, max_dist_m=max_dist_m
        )

        # Save
        out_path = os.path.join(processed_path, city_name, "metadata.gpkg")
        aggregate_and_save(matches, building_gdf, entrance_gdf, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
