# nctempps/validation.py

from __future__ import annotations
import os
from typing import List, Sequence, Tuple
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterstats import zonal_stats
from nctempps.models.spatial_uncertainty import corr_with_spatial_uncertainty


def load_noaa_traversal(noaa_root: str, city_name: str, noaa_time_slot: str) -> gpd.GeoDataFrame:
    '''Loads NOAA traversal shapefile for given city and time slot.'''
    noaa_trav_locs = "raleigh_durham" if city_name in ["durham", "raleigh"] else city_name
    trav_path = os.path.join(noaa_root, f"noaa_{city_name}", f"{noaa_trav_locs}_traversals", f"{noaa_time_slot}_trav.shp")
    trav = gpd.read_file(trav_path)
    # check for temp_f or t_f column and convert to temp_C
    if "t_f" in trav.columns:
        trav["temp_C"] = (trav["t_f"] - 32) * 5.0 / 9.0
    if "temp_f" in trav.columns:
        trav["temp_C"] = (trav["temp_f"] - 32) * 5.0 / 9.0
    return trav


def aggregate_trav_to_units(trav: gpd.GeoDataFrame, areal_units: gpd.GeoDataFrame, county_prefixes: List[str], temp_col: str = "temp_C",
) -> gpd.GeoDataFrame:
    '''Aggregates NOAA traversal temperatures to areal units (e.g., census tracts) within specified counties.'''

    if temp_col not in trav.columns:
        raise ValueError(f"Temperature column '{temp_col}' not found in traversal data")

    # filter areal_units to county/city
    au = areal_units.copy()
    au["GEOID"] = au["GEOID"].astype(str)
    keep = au["GEOID"].str.startswith(tuple(county_prefixes))
    city_units = au.loc[keep].copy().to_crs(trav.crs)

    trav_pts = trav[[temp_col, "geometry"]].copy()
    joined = gpd.sjoin(trav_pts, city_units[["GEOID", "geometry"]], how="inner", predicate="within")
    noaa_by_unit = (
        joined.groupby("GEOID")[temp_col]
        .mean()
        .reset_index()
        .rename(columns={temp_col: "noaa_trav_temp"})
    )

    return city_units.merge(noaa_by_unit, on="GEOID", how="left")


def load_aggregate_noaa_raster(noaa_root: str, city_name: str, noaa_time_slot: str, areal_units: gpd.GeoDataFrame, temp_col: str = "temp_C") -> gpd.GeoDataFrame:
    """Aggregate NOAA GeoTIFF to areal units using zonal mean."""

    noaa_trav_locs = "raleigh_durham" if city_name in ["durham", "raleigh"] else city_name
    tif_path = os.path.join(noaa_root, f"noaa_{city_name}", f"{noaa_trav_locs}_rasters", f"{noaa_time_slot}.tif")
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"NOAA raster not found: {tif_path}")
    if "GEOID" not in areal_units.columns:
        raise ValueError("areal_units must contain a 'GEOID' column")

    gdf = areal_units.copy()
    gdf["GEOID"] = gdf["GEOID"].astype(str)

    with rasterio.open(tif_path) as src:
        if gdf.crs is None:
            raise ValueError("areal_units has no CRS set")
        if src.crs is None:
            raise ValueError("Raster has no CRS metadata")
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        nodata = src.nodata

        # zonal stats: mean value per polygon
        zs = zonal_stats(
            vectors=gdf.geometry,
            raster=tif_path,
            stats=["mean"],
            nodata=nodata,
            all_touched=False,
            geojson_out=False,
        )

    means = np.array([d["mean"] for d in zs], dtype=float)
    means_C = (means - 32) * 5.0 / 9.0
    gdf[temp_col] = means_C
    gdf = gdf.to_crs(areal_units.crs)
    gdf = gdf.rename(columns={temp_col: "noaa_raster_temp"})

    return gdf


def load_svgpr_predictions(data_dir: str, city_name: str, census_group_abbrv: str, m: int, wu_date: str, time_slot: str,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    '''
    Loads SVGPR and SVGPR+PS predictions for given city, census group, model, date, and time slot.
    Returns two GeoDataFrames: (svgpr_gdf, svgpr_ps_gdf)
    '''
    svgpr_path = os.path.join(data_dir, "ps_on_nc", city_name, f"{city_name}_svgpr_{census_group_abbrv}_{m}_{wu_date}_{time_slot}.shp")
    svgpr_ps_path = os.path.join(data_dir, "ps_on_nc", city_name, f"{city_name}_svgpr_ps_{census_group_abbrv}_{m}_{wu_date}_{time_slot}.shp")
    svgpr_gdf = gpd.read_file(svgpr_path)
    svgpr_ps_gdf = gpd.read_file(svgpr_ps_path)

    for g in (svgpr_gdf, svgpr_ps_gdf):
        g["GEOID"] = g["GEOID"].astype(str)
        
    svgpr_gdf = svgpr_gdf.rename(columns={"pred_temp": "svgpr"})
    svgpr_ps_gdf = svgpr_ps_gdf.rename(columns={"pred_temp": "svgpr_ps"})
    return svgpr_gdf, svgpr_ps_gdf


TruthTarget = Tuple[str, str, str]


def print_spatial_uncertainty_corrs(
    *,
    df_compare: pd.DataFrame,
    city_units_truth: gpd.GeoDataFrame,
    census_group_type: str,
    city_name: str,
    time_slot: str,
    latent_in_cov: bool,
    truth_source: str,
    truth_targets: Sequence[TruthTarget],  # REQUIRED
    max_dist: float = 10_000.0,
    bin_width: float = 1_000.0,
) -> dict:
    """
    Prints correlation summary and returns metrics dict.

    Requires:
      - `city_units_truth` contains geometry + truth cols listed in truth_targets
      - `df_compare` contains GEOID + prediction cols
    """
    df_geo = None

    try:
        if truth_targets is None or len(truth_targets) == 0:
            raise ValueError("truth_targets is required and cannot be empty.")

        pred_cols = ["GEOID", "svgpr", "svgpr_ps", "exact_gp"]
        missing = [c for c in pred_cols if c not in df_compare.columns]
        if missing:
            raise KeyError(f"df_compare missing columns {missing}. Has: {df_compare.columns.tolist()}")

        if "geometry" not in city_units_truth.columns:
            raise KeyError("city_units_truth must contain a 'geometry' column.")

        city_metric = city_units_truth.drop(columns=["exact_gp", "svgpr", "svgpr_ps"], errors="ignore").copy()
        city_metric = city_metric.to_crs("EPSG:32119")
        city_metric["GEOID"] = city_metric["GEOID"].astype(str)

        df_compare = df_compare.copy()
        df_compare["GEOID"] = df_compare["GEOID"].astype(str)

        df_geo = city_metric.merge(df_compare[pred_cols], on="GEOID", how="inner")
        df_geo = gpd.GeoDataFrame(df_geo, geometry="geometry", crs=city_metric.crs)

        missing_truth = [ycol for _, ycol, _ in truth_targets if ycol not in df_geo.columns]
        if missing_truth:
            raise KeyError(
                f"Missing truth columns in merged df_geo: {missing_truth}. "
                f"Has: {df_geo.columns.tolist()}"
            )

        centroids = df_geo.geometry.centroid
        coords_m = np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])

        specs = [
            ("svgpr",    "svgpr",    "SVGPR"),
            ("svgpr_ps", "svgpr_ps", "SVGPR+PS"),
            ("exact_gp", "exact_gp", "LGCP+GP" if latent_in_cov else "ExactGP"),
        ]

        print(f"\n{census_group_type} data ({city_name}) - {time_slot} (n={len(df_geo)}):")

        out: Dict[str, Dict[str, Any]] = {}

        for tkey, ycol, label in truth_targets:
            y = df_geo[ycol].to_numpy()
            out[tkey] = {}

            for key, pred_col, model_label in specs:
                m = corr_with_spatial_uncertainty(
                    df_geo[pred_col].to_numpy(),
                    y,
                    coords_m,
                    max_dist=max_dist,
                    bin_width=bin_width,
                )
                out[tkey][key] = m
                print(
                    f"{model_label} vs {truth_source} {label}: "
                    f"r = {m['r']:.2f} ± {m['r_sd']:.2f} "
                    f"(95% CI [{m['r_lo']:.2f}, {m['r_hi']:.2f}]), "
                    f"p = {m['p']:.2e}, M_eff = {m['M_eff']:.1f}"
                )

            print(
                f"SVGPR+PS - SVGPR ({truth_source} {label}):  "
                f"Δr = {out[tkey]['svgpr_ps']['r'] - out[tkey]['svgpr']['r']:.2f}"
            )
            print(
                f"{('LGCP+GP' if latent_in_cov else 'ExactGP')} - SVGPR ({truth_source} {label}):  "
                f"Δr = {out[tkey]['exact_gp']['r'] - out[tkey]['svgpr']['r']:.2f}\n"
            )

        # Backward compatible return shape
        if len(truth_targets) == 1:
            return out[truth_targets[0][0]]
        return out

    except Exception as e:
        print(f"print_spatial_uncertainty_corrs: {e}")
        raise