'''
This script processes weather station data, spatially joins it with areal units (tracts or block groups),
aggregates temperature data, adds population and elevation information, computes centroids, and extracts night-time
land surface temperature from a raster file. The final output is saved as both a CSV and a shapefile.
'''

import argparse
from pathlib import Path
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from rasterstats import zonal_stats
import tempfile
import os


# Coordinate reference systems:
WORK_CRS = "EPSG:32119"
GEOG_CRS = "EPSG:4326"

from dataclasses import dataclass

def spatial_join_and_aggregate(
    gdf_points,
    areal_units,
    *,
    station_col: str,
    temp_col: str,
    temp_out_col: str,
    geoid_col: str = "GEOID",
):
    if areal_units.crs != gdf_points.crs:
        areal_units = areal_units.to_crs(gdf_points.crs)

    pts = gpd.sjoin(
        gdf_points,
        areal_units[[geoid_col, "geometry"]],
        how="left",
        predicate="within",
    )

    pts = pts.dropna(subset=[station_col, temp_col, geoid_col])

    per_station = (
        pts.groupby([geoid_col, station_col], dropna=False)[temp_col]
           .median()
           .reset_index(name="station_median")
    )

    per_geoid = (
        per_station.groupby(geoid_col, dropna=False)
                   .agg(
                       **{temp_out_col: ("station_median", "median")},
                       count=(station_col, "nunique")
                   )
                   .reset_index()
    )

    out = areal_units.merge(per_geoid, on=geoid_col, how="left")
    out["count"] = out["count"].fillna(0).astype("Int64")
    return out

def add_lst_from_table(areal_units, *, cov_root: str, year: int, month: int, slot: str):
    cov_root = Path(cov_root)
    slot = slot.lower()
    t_s_folder = "t_s_6am" if slot == "6am" else "t_s_4pm"
    lst_path = cov_root / "modis_lst" / "block_group" / t_s_folder / f"nc_avg_{year}_{month:02d}.csv"
    lst = pd.read_csv(lst_path, dtype={"GEOID": str})
    t_s_col = f"t_s_{slot}"
    if t_s_col not in lst.columns:
        raise ValueError(f"Missing {t_s_col} in {lst_path}")
    areal_units = areal_units.drop(columns=[t_s_col], errors="ignore")
    return areal_units.merge(lst[["GEOID", t_s_col]], on="GEOID", how="left")

def add_population_from_table(areal_units, *, cov_root: str, year: int):
    cov_root = Path(cov_root)
    pop_path = cov_root / "pop_by_bg" / str(year) / "pop_by_bg.csv"
    if not pop_path.exists() and year == 2025:
        pop_path = cov_root / "pop_by_bg" / "2024" / "pop_by_bg.csv"
    pop = pd.read_csv(pop_path, dtype={"GEOID": str})
    if "population" not in pop.columns:
        raise ValueError(f"{pop_path} missing 'population' col")
    areal_units = areal_units.drop(columns=["population"], errors="ignore")
    return areal_units.merge(pop[["GEOID", "population"]], on="GEOID", how="left")

def add_elevation_from_table(areal_units, *, cov_root: str):
    cov_root = Path(cov_root)
    elev_path = cov_root / "elevation_by_bg" / "data.csv"
    elev = pd.read_csv(elev_path, dtype={"GEOID": str})
    if "elevation" not in elev.columns:
        raise ValueError(f"{elev_path} missing 'elevation'")
    areal_units = areal_units.drop(columns=["elevation"], errors="ignore")
    return areal_units.merge(elev[["GEOID", "elevation"]], on="GEOID", how="left")

def add_centroids(areal_units):
    projected = areal_units.to_crs(WORK_CRS)
    centr_m = projected.geometry.centroid
    projected["x_m"] = centr_m.x
    projected["y_m"] = centr_m.y
    cent_ll = gpd.GeoSeries(centr_m, crs=WORK_CRS).to_crs(GEOG_CRS)
    projected["lon"] = cent_ll.x
    projected["lat"] = cent_ll.y
    return projected

@dataclass
class ArealizeOutputs:
    geo_csv: str
    elev_csv: str
    shp_file: str

def process_one_day_slot(
    wu_csv: str,
    time_slot: str,
    year: int,
    month: int,
    cov_root: str,
    out_root: str,
    wu_date_folder: str,
    *,
    station_col: str = "station_id",
    lat_col: str = "lat",
    lon_col: str = "lon",
    areal_unit_prefix: str = "nc_bg",
) -> ArealizeOutputs:
    
    time_slot = time_slot.lower()
    if time_slot not in {"6am", "4pm"}:
        raise ValueError(f"time_slot must be '6am' or '4pm', got {time_slot}")
    
    df = pd.read_csv(wu_csv)
    hour_map = {
        "6am": "h06",
        "4pm": "h16",
    }
    hour_code = hour_map[time_slot]
    temp_col = f"tempAvg_{hour_code}_median_fullhour"
    if temp_col not in df.columns:
        raise ValueError(f"{wu_csv} missing expected column {temp_col}")

    missing = [c for c in (lat_col, lon_col, temp_col, station_col) if c not in df.columns]
    if missing:
        raise ValueError(f"{wu_csv} missing columns: {missing}. Got: {df.columns.tolist()}")

    df = df.dropna(subset=[lat_col, lon_col, temp_col, station_col])

    gdf_pts = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs=GEOG_CRS,
    ).to_crs(WORK_CRS)

    elev_shp = Path(cov_root) / "elevation_by_bg" / "data.shp"
    areal_units = gpd.read_file(elev_shp)[["GEOID", "ALAND", "AWATER", "geometry"]].copy()
    areal_units["GEOID"] = areal_units["GEOID"].astype(str)
    if areal_units["GEOID"].str.len().nunique() != 1:
        areal_units["GEOID"] = areal_units["GEOID"].str.zfill(12)

    areal_units = areal_units.to_crs(WORK_CRS)

    temp_out_col = f"{time_slot}_temp"
    areal_units = spatial_join_and_aggregate(
        gdf_pts,
        areal_units,
        station_col=station_col,
        temp_col=temp_col,
        temp_out_col=temp_out_col,
        geoid_col="GEOID",
    )

    areal_units = add_population_from_table(areal_units, cov_root=cov_root, year=year)
    areal_units = add_elevation_from_table(areal_units, cov_root=cov_root)
    areal_units = add_lst_from_table(areal_units, cov_root=cov_root, year=year, month=month, slot=time_slot)

    areal_units["count"] = areal_units["count"].fillna(0).astype(int)

    out_dir = Path(out_root) / "data" / wu_date_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    geo_csv = out_dir / f"{areal_unit_prefix}_temp_pop_{time_slot}.csv"
    shp_file = out_dir / f"{areal_unit_prefix}_temp_pop_{time_slot}.shp"
    keep_geo = ["GEOID", "ALAND", "AWATER", temp_out_col, "count", "population"]
    
    geo_out = areal_units[keep_geo + ["geometry"]].copy()
    geo_out.to_csv(geo_csv, index=False)
    geo_out.to_file(shp_file)

    t_s_col = f"t_s_{time_slot}"
    elev_csv = out_dir / f"{areal_unit_prefix}_elev_{t_s_col}_{time_slot}.csv"
    elev_df = areal_units[["GEOID", "elevation", t_s_col]].copy()
    elev_df.to_csv(elev_csv, index=False)

    return ArealizeOutputs(str(geo_csv), str(elev_csv), str(shp_file))