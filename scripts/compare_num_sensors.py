#!/usr/bin/env python3
"""
Compare different numbers of WU sensors used in the LGCP(+GP) experiment.

- Starts from a median-aggregated areal data for all of NC, in a specific hour time slot of a specific day.
- For each sensor-count setting, randomly (but deterministically) subsamples stations,
  writes a subset CSV, runs the experiment, and logs metrics to an output CSV.

Author: Ellie Kim
"""

from __future__ import annotations
import argparse
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os
from lgcp_gp import run_lgcp_exactgp_experiment, county_prefixes_for_city


SLOT_WINDOWS = {
    # local NC time (America/New_York)
    "morn": (6, 7),
    "af":   (15, 16),
    "eve":  (19, 20),
}


def make_trial_seed(base_seed: int, n_sensors: int, trial: int, k_trials: int) -> int:
    """
    Create a unique random seed for this (n_sensors, trial) combination,
    based on a base seed.
    This helps ensure different random subsamples across trials and n_sensors values.
    NB: designed for 0-based trial index.

    Inputs:
    - base_seed: base integer seed
    - n_sensors: number of sensors (int)
    - trial: trial index (int)
    Returns:
    - unique integer seed for this (n_sensors, trial)
    """
    # deterministic and spaced out; avoids collisions across (n, trial)
    return int(base_seed + int(n_sensors) * k_trials + int(trial))


def print_single_result(i: int, total: int, row: Dict[str, Any]) -> None:
    def fmt(x: Any) -> str:
        try:
            return f"{float(x):.4f}"
        except Exception:
            return "nan"

    def norm_city(x: Any) -> str:
        return str(x).strip().lower().replace(" ", "_").replace("-", "_")

    city = norm_city(row.get("city_name", ""))
    noaa_mode = str(row.get("noaa_data_type", "")).strip().lower()

    if city == "chapel_hill":
        corr_str = f"heatwatch={fmt(row.get('heatwatch_corr'))}"
    else:
        if noaa_mode == "trav":
            corr_str = f"noaa_trav={fmt(row.get('noaa_corr_trav'))}"
        elif noaa_mode == "raster":
            corr_str = f"noaa_raster={fmt(row.get('noaa_corr_raster'))}"
        else:  # "trav_raster" (or anything else)
            corr_str = (
                f"noaa_trav={fmt(row.get('noaa_corr_trav'))}, "
                f"noaa_raster={fmt(row.get('noaa_corr_raster'))}"
            )

    print(
        f"[{i}/{total}] {row.get('exp_name', '<no exp_name>')}: "
        f"n_sensors={row.get('n_sensors', 'NA')}, "
        f"{corr_str}, "
        f"loo={row.get('loo_pseudoll_per_point', float('nan')):.7g}, "
        f"gp_sec={row.get('gp_train_seconds', float('nan')):.1f}"
    )


def filter_data_to_time_slot(df: pd.DataFrame, time_slot: str, metric_col: str, last10: bool) -> pd.DataFrame:
    """
    Filters all station observations for this specific day to the desired time slot (e.g., 14:00-15:00 UTC).
    - May contain multiple observations per station in the time slot.
    - If last10 is True, further filters to only the last 10 minutes of the hour time slot.
    """
    start_h, end_h = SLOT_WINDOWS[time_slot]

    # Basic: whole hour window [start_h, end_h)
    mask_slot = df["hour_local"].between(start_h, end_h, inclusive="left")
    df_slot = df.loc[mask_slot].copy()

    print(f"{time_slot}: kept {len(df_slot)} / {len(df)} observations")

    # optionally, filter to last 10 minutes of the hour
    if last10:
        df_slot = df_slot[df_slot["minute_local"] >= 50].copy()
        print(f"{time_slot} last10: kept {len(df_slot)} observations from time slot")

    # clean up data before returning it:
    df_slot[metric_col] = pd.to_numeric(df_slot[metric_col], errors="coerce") # optional: make sure temp is numeric
    df_slot = df_slot.dropna(subset=[metric_col, "lat", "lon"]).copy() # drop NaNs

    return df_slot


def get_median_per_station(df_slot: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """
    Get median temperature value per station, if there are multiple observations per station in the time slot.
    Also get count of observations per station, plus first/last observation time.
    """
    station_summary = (
        df_slot
        .groupby("station_id", as_index=False)
        .agg(
            n_obs=(metric_col, "size"),
            temp_median=(metric_col, "median"),
            lat=("lat", "first"),
            lon=("lon", "first"),
            first_time=("obsTimeLocal", "min"),
            last_time=("obsTimeLocal", "max"),
        )
    )
    print(f"Aggregated to {len(station_summary)} stations (from {len(df_slot)} observations)")
    return station_summary


def split_by_station_id(station_summary, n_keep, seed=0):
    """
    Helper function for get_randomized_areal_data.
    This function randomly keeps n_keep sensors (rows) and holds out the rest.

    Returns:
    - kept: DataFrame of kept stations (n_keep rows)
    - heldout: DataFrame of held-out stations (len(station_summary) - n_keep rows)
    """
    rng = np.random.default_rng(seed)
    if n_keep > len(station_summary):
        raise ValueError("n_keep too large")
    keep_idx = rng.choice(station_summary.index.to_numpy(), size=n_keep, replace=False)
    kept = station_summary.loc[keep_idx].copy()
    heldout = station_summary.drop(index=keep_idx).copy()
    return kept, heldout


def aggregate_points_to_polys(gdf_pts, polys, value_col, geoid_col="GEOID"):
    """
    Other helper function for get_randomized_areal_data.

    This function aggregates a point GeoDataFrame to polygon GeoDataFrame by taking median of value_col within each polygon.
    - value_col: column in gdf_pts to aggregate (e.g., "temp_median")

    Returns a GeoDataFrame with the same polygons, plus columns:
    - n_obs: number of points within each polygon
    - median_val: median of value_col within each polygon
    """
    # inner = only points that land in a polygon
    joined = gpd.sjoin(
        gdf_pts,
        polys[[geoid_col, "geometry"]],
        how="inner",
        predicate="within",
    )

    agg = (joined.groupby(geoid_col, as_index=False)
                 .agg(
                     n_obs=(value_col, "size"),
                     median_val=(value_col, "median"),
                 ))
    polys_agg = polys.merge(agg, on=geoid_col, how="left")
    return polys_agg


def kept_vs_heldout_aggregated_plot(n_keep, polys_kept, polys_all, county_prefix, out_dir, trial) -> str:
    """
    Yet another helper function for get_randomized_areal_data.

    This function makes a side-by-side plot comparing the aggregated median values
    for the kept stations vs all stations, within the specified county (by FIPS prefix).
    
    Returns the path to the saved figure.
    """
    # filter to desired county (by GEOID prefix)
    mask_kept = polys_kept["GEOID"].str.startswith(tuple(county_prefix))
    mask_all  = polys_all["GEOID"].str.startswith(tuple(county_prefix))

    polys_kept_c = polys_kept.loc[mask_kept].copy()
    polys_all_c  = polys_all.loc[mask_all].copy()

    print(f"County polygons: {len(polys_kept_c)} (kept), {len(polys_all_c)} (all)")
    print(f"County polygons with data (kept): {polys_kept_c['median_val'].notna().sum()} / {len(polys_kept_c)}")
    print(f"County polygons with data (all):  {polys_all_c['median_val'].notna().sum()} / {len(polys_all_c)}")

    # shared color scale across both county plots
    vals = pd.concat([polys_kept_c["median_val"], polys_all_c["median_val"]], axis=0)
    vmin, vmax = np.nanpercentile(vals.to_numpy(), [2, 98])  # or np.nanmin/np.nanmax

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

    polys_kept_c.plot(
        column="median_val",
        ax=axes[0],
        vmin=vmin, vmax=vmax,
        legend=True,
        edgecolor="black",
        linewidth=0.2,
        missing_kwds={"color": "lightgrey", "label": "No data"},
    )
    axes[0].set_title(f"Kept sensors (n={n_keep})\nMedian temp per polygon")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].grid(False)

    polys_all_c.plot(
        column="median_val",
        ax=axes[1],
        vmin=vmin, vmax=vmax,
        legend=True,
        edgecolor="black",
        linewidth=0.2,
        missing_kwds={"color": "lightgrey", "label": "No data"},
    )
    axes[1].set_title("All sensors\nMedian temp per polygon")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    axes[1].grid(False)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"kept_vs_all_aggregated_for_{n_keep}_sensors_trial{trial}.png")
    plt.savefig(fig_path)
    plt.close()

    return fig_path


def get_randomized_areal_data(
    station_summary: pd.DataFrame,
    n_keep: int,
    wu_date: str,
    noaa_date: str,
    time_slot: str,
    data_root: str,
    out_dir: str,
    city_name: str, # e.g. "charlotte", "durham", etc.
    census_group_type: str = "block_group", # 'tract' or 'block_group' --> determines which areal_shp file to use
    *,
    trial: int = 0,
    trial_seed: int = 42,
    save_maps: bool = True
) -> str:
    """
    Randomly subsample the station_summary to n_sensors stations (visualizes the kept vs held-out stations), 
    then aggregate station data by taking median per block group / census tract,
    then write aggregated areal data to CSV to get called by main LGCP+GP experiment function.

    Returns the path to the saved CSV file.

    Inputs:
    - station_summary: DataFrame with per-station median data (from get_median_per_station)
    - n_keep: number of stations to keep (subsample)
    - wu_date, noaa_date, time_slot: identifiers for naming output files
    - data_root: root path to deep_ps repo
    - out_dir: directory to save output files (should be diff for each trial and n_sensors setting)
    - city_name: e.g. "charlotte", "durham", etc.
    - census_group_type: 'tract' or 'block_group' --> determines which areal_shp file to use
    - trial: trial index (for multiple trials per n_sensors)
    - trial_seed: random seed for this trial
    """

    # kept, heldout = split_by_station_id(station_summary, n_keep=n_keep, seed=42 + int(n_keep))
    # before, the random seed was dependent on the n_sensors value, which could lead to collisions across trials
    # now, each trial gets a different random seed, independent of n_sensors value:
    kept, heldout = split_by_station_id(station_summary, n_keep=n_keep, seed=trial_seed)

    if save_maps:
        # create a figure with 2 subplots: kept on left, heldout on right
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(kept["lon"], kept["lat"], c="green", marker="o", label="kept")
        plt.title(f"Kept Sensors (n={n_keep})")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.grid()
        plt.subplot(1, 2, 2)
        plt.scatter(heldout["lon"], heldout["lat"], c="lightgray", marker="x", label="held-out")
        plt.title(f"Held-out Sensors (n={len(heldout)})")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        # save the figure showing held-out vs kept sensors
        sensors_fig_path = os.path.join(out_dir, f"heldout_vs_kept_{city_name}_{wu_date}_{noaa_date}_{time_slot}_n{n_keep}_trial{trial}.png")
        plt.savefig(sensors_fig_path)
        plt.close()

    # get shp file (either in census tracts or block groups)
    census_group_tag = 'tract' if census_group_type == 'tract' else 'bg'
    areal_shp = os.path.join(data_root, "data", wu_date, f"nc_{census_group_tag}_temp_pop_{time_slot}.shp")
    print("areal_shp file cols: ", gpd.read_file(areal_shp).columns.tolist())
    polys = gpd.read_file(areal_shp).to_crs("EPSG:4326") # load polygons from areal_shp + ensure GEOID is string
    polys["GEOID"] = polys["GEOID"].astype(str)

    # make points GeoDataFrames, points are from per-station medians
    gdf_pts_kept = gpd.GeoDataFrame(
        kept.copy(),
        geometry=gpd.points_from_xy(kept["lon"], kept["lat"]),
        crs="EPSG:4326"
    )
    gdf_pts_all = gpd.GeoDataFrame(
        station_summary.copy(),
        geometry=gpd.points_from_xy(station_summary["lon"], station_summary["lat"]),
        crs="EPSG:4326"
    )
    print(f"{len(gdf_pts_kept)} kept stations out of {len(gdf_pts_all)} total unique stations")

    # Aggregate to polygons (take median of station medians within each polygon)
    polys_kept = aggregate_points_to_polys(gdf_pts_kept, polys, value_col="temp_median")
    polys_all  = aggregate_points_to_polys(gdf_pts_all,  polys, value_col="temp_median")
    print(f"Polygons with data (kept): {polys_kept['median_val'].notna().sum()} / {len(polys_kept)}")
    print(f"Polygons with data (all):  {polys_all['median_val'].notna().sum()} / {len(polys_all)}")

    # make kept-vs-all aggregated plot for this n_keep
    if save_maps:
        county_prefixes = county_prefixes_for_city(city_name)
        fig_path = kept_vs_heldout_aggregated_plot(n_keep, polys_kept, polys_all, county_prefixes, out_dir, trial)
        print(f"Saved kept-vs-all aggregated plot to {fig_path}")

    # prepare polys_kept for saving to CSV
    polys_kept = polys_kept[["GEOID", "geometry", "n_obs", "median_val", "population"]]
    polys_kept = polys_kept.rename(columns={"median_val": f"{time_slot}_temp"}) # e.g. morn_temp, af_temp, eve_temp

    # save to CSV and SHP files so these can be read by our regular LGCP+GP experiment function
    # areal_csv_path = os.path.join(data_root, "data", wu_date, "sensor_data", f"kept_{n_keep}_sensors_nc_{census_group_tag}_temp_pop_{time_slot}.csv")
    # os.makedirs(os.path.dirname(areal_csv_path), exist_ok=True)
    # areal_shp_path = os.path.join(data_root, "data", wu_date, "sensor_data", f"kept_{n_keep}_sensors_nc_{census_group_tag}_temp_pop_{time_slot}.shp")
    
    areal_csv_path = os.path.join(
        out_dir,
        f"kept_n{n_keep}_sensors_trial{trial}_{census_group_tag}_temp_pop_{time_slot}.csv"
    )
    areal_shp_path = os.path.join(
        out_dir,
        f"kept_n{n_keep}_sensors_trial{trial}_{census_group_tag}_temp_pop_{time_slot}.shp"
    )
    # shp includes geometry; csv does not
    polys_kept.to_file(areal_shp_path)
    polys_kept.drop(columns=["geometry"]).to_csv(areal_csv_path, index=False)
    print(f"Saved aggregated areal data for kept sensors to {areal_csv_path}")
    return areal_csv_path


def summarize_trials_by_n_sensors(
    results: list[Dict[str, Any]],
    *,
    corr_cols: list[str], # e.g. ['noaa_corr_trav', 'noaa_corr_raster'] as returned by run_lgcp_exactgp_experiment function
) -> pd.DataFrame:
    """
    Groups trial-level results by n_sensors and computes mean/std for specified corr columns.
    
    Inputs:
    - results: list of result dicts, each containing 'n_sensors' and correlation columns
    - corr_cols: list of correlation column names to summarize (e.g., ['noaa_corr_trav', 'noaa_corr_raster'])

    Returns a DataFrame with columns:
       n_sensors,
       <col>_mean, <col>_std,
       n_trials
    """
    df = pd.DataFrame(results).copy()
    if "n_sensors" not in df.columns:
        raise ValueError("Expected 'n_sensors' in results rows.")
    
    df["n_sensors"] = pd.to_numeric(df["n_sensors"], errors="coerce")
    df = df.dropna(subset=["n_sensors"]).copy()
    df["n_sensors"] = df["n_sensors"].astype(int)

    summary_rows = []
    for n_sensors, group in df.groupby("n_sensors"):
        summary_row = {"n_sensors": n_sensors, "n_trials": len(group)}
        for col in corr_cols:
            if col in group.columns:
                vals = pd.to_numeric(group[col], errors="coerce").dropna()
                if not vals.empty:
                    summary_row[f"{col}_mean"] = vals.mean()
                    # summary_row[f"{col}_std"] = vals.std()
                    summary_row[f"{col}_std"] = vals.std(ddof=0)
                else:
                    summary_row[f"{col}_mean"] = float("nan")
                    summary_row[f"{col}_std"] = float("nan")
            else:
                summary_row[f"{col}_mean"] = float("nan")
                summary_row[f"{col}_std"] = float("nan")
        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("n_sensors").reset_index(drop=True)

    return summary_df


def plot_corr_vs_n_sensors(
    results: list[Dict[str, Any]],
    *,
    out_dir: str,
    wu_date: str,
    city_name: str,
    time_slot: str,
    census_group_type: str,
    noaa_data_type: str,
    exp_name_hint: str = "lgcp_gp",
) -> str:
    # If chapel hill, plot Heatwatch; else plot NOAA as before.
    if city_name.lower() == "chapel_hill":
        corr_cols = ["heatwatch_corr"]
        summary = summarize_trials_by_n_sensors(results, corr_cols=corr_cols)

        fig = plt.figure(figsize=(10, 6))
        ax = plt.gca()
        x = summary["n_sensors"].to_numpy(dtype=float)
        y = summary["heatwatch_corr_mean"].to_numpy(dtype=float)
        keep = ~np.isnan(y)
        if keep.sum() < 1:
            raise ValueError("No usable Heatwatch correlation values to plot.")
        ax.plot(x[keep], y[keep], marker="o", linestyle="-", label="UNC Heatwatch")
        ax.set_xlabel("Number of sensors")
        ax.set_ylabel("Correlation with Heatwatch")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        fname = (
            f"{wu_date}_{city_name}_{time_slot}_{census_group_type}_"
            f"{exp_name_hint}_heatwatch_corr_vs_n_sensors_mean_over_trials.png"
        )
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"[PLOT] Saved Heatwatch corr vs n_sensors plot to: {out_path}")
        return out_path

    # Otherwise use your existing NOAA plotting function
    return plot_noaa_corr_vs_n_sensors(
        results,
        out_dir=out_dir,
        wu_date=wu_date,
        city_name=city_name,
        time_slot=time_slot,
        census_group_type=census_group_type,
        noaa_data_type=noaa_data_type,
        exp_name_hint=exp_name_hint,
    )


def plot_noaa_corr_vs_n_sensors(
    results: list[Dict[str, Any]],
    *,
    out_dir: str,
    wu_date: str,
    city_name: str,
    time_slot: str,
    census_group_type: str,
    noaa_data_type: str,
    exp_name_hint: str = "lgcp_gp",
) -> str:
    """
    Plot mean NOAA correlation vs n_sensors, averaging across trials for each n_sensors.

    - Traversal series uses: noaa_corr_trav
    - Raster series uses:    noaa_corr_raster

    Saves a PNG and returns its path.
    """
    if not results:
        raise ValueError("No results to plot.")

    # Decide which series we want to plot given the mode
    want_trav = noaa_data_type in ("trav", "trav_raster")
    want_raster = noaa_data_type in ("raster", "trav_raster")

    # Columns we will look for in results
    trav_col = "noaa_corr_trav"
    raster_col = "noaa_corr_raster"

    corr_cols = []
    if want_trav:
        corr_cols.append(trav_col)
    if want_raster:
        corr_cols.append(raster_col)
    
    # Summarize across trials per n_sensors (call helper fn)
    summary = summarize_trials_by_n_sensors(results, corr_cols=corr_cols)

    # Prepare plot
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()

    x = summary["n_sensors"].to_numpy(dtype=float)
    plotted_any = False

    # Traversal: mean across trials (same number of trials k per n_sensors)
    if want_trav and f"{trav_col}_mean" in summary.columns:
        y = summary[f"{trav_col}_mean"].to_numpy(dtype=float)
        keep = ~np.isnan(y)
        if keep.sum() >= 1:
            ax.plot(x[keep], y[keep], marker="o", linestyle="-", label="NOAA Traversal")
            plotted_any = True

    # Raster: mean across trials
    if want_raster and f"{raster_col}_mean" in summary.columns:
        y = summary[f"{raster_col}_mean"].to_numpy(dtype=float)
        keep = ~np.isnan(y)
        if keep.sum() >= 1:
            ax.plot(x[keep], y[keep], marker="o", linestyle="-", label="NOAA Raster")
            plotted_any = True

    if not plotted_any:
        raise ValueError(
            "No usable NOAA correlation values to plot after averaging.\n"
            f"noaa_data_type={noaa_data_type}\n"
            f"summary columns={list(summary.columns)}"
        )

    ax.set_xlabel("Number of sensors")
    ax.set_ylabel("Correlation with NOAA")
    # ax.errorbar(x[keep], y[keep], yerr=summary[f"{trav_col}_std"].to_numpy(dtype=float)[keep], fmt="o-", label="NOAA traversal (mean ± std)" if want_trav else "")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    fname = (
        f"{wu_date}_{city_name}_{time_slot}_{census_group_type}_"
        f"{exp_name_hint}_noaa_corr_vs_n_sensors_{noaa_data_type}_mean_over_trials.png"
    )
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[PLOT] Saved NOAA corr vs n_sensors plot to: {out_path}")
    return out_path


# # TODO: Write a helper function that takes the final output csv of all trials and n_sensors,
# # and plots an aggregated version of the predictions for all k trials for each n_sensors value.
# # The result should be a set of maps (one per n_sensors) showing the mean temp prediction
# # across all trials at each spatial location.

# def plot_aggregated_predictions(results: pd.DataFrame, n_sensors: List[int], out_dir: str) -> None:
#     """
#     NOTE: THIS FUNCTION IS NOT COMPLETE AND STILL NEEDS TO BE IMPLEMENTED.
#     The intended purpose is to get aggregative *qualitative* plots of the temp predictions for each n_sensors value,
#     averaged across all trials.
#     """
#     for n in n_sensors:
#         # Filter results for the current n_sensors value
#         subset = results[results["n_sensors"] == n]

#         if subset.empty:
#             continue

#         # Compute mean temp prediction across trials
#         mean_prediction = subset.groupby("spatial_location")["temp_prediction"].mean()

#         # Plotting code here (e.g., using matplotlib or another library)
#         plt.figure(figsize=(10, 6))
#         mean_prediction.plot(kind="bar")
#         plt.title(f"Mean Temp Prediction (n_sensors={n})")
#         plt.xlabel("Spatial Location")
#         plt.ylabel("Mean Temp Prediction")
#         plt.grid(True)

#         # Save the figure
#         plt.tight_layout()
#         plt.savefig(os.path.join(out_dir, f"mean_temp_prediction_n_sensors_{n}.png"), dpi=200)
#         plt.close()

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep ExactGP latent kernel/lengthscale configs via exp_json.")
    p.add_argument("--data_root", type=str, required=True,
               help="Path to deep_ps repo root")
    p.add_argument("--noaa_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="./results")

    p.add_argument("--wu_date", type=str, default="jul23_2021")
    p.add_argument("--last10", action="store_true", default=False, help="Whether to only use the last 10 minutes of the hour time slot.")
    p.add_argument("--noaa_date", type=str, default="20210723")
    p.add_argument("--metric_col", type=str, default="tempAvg")
    p.add_argument("--census_group_type", type=str, choices=["tract", "block_group"], default="block_group")
    p.add_argument("--city_name", type=str, default="durham")
    p.add_argument("--time_slot", type=str, choices=["morn", "af", "eve"], default="eve")

    p.add_argument("--m_inducing", type=int, default=800)
    p.add_argument("--inducing_type", type=str, choices=["kmeans", "grid", "random", "min_separation"], default="min_separation")
    p.add_argument("--lgcp_iters", type=int, default=500)
    p.add_argument("--lgcp_lr", type=float, default=0.01)
    p.add_argument("--lgcp_particles", type=int, default=32)
    p.add_argument("--lgcp_lrd", type=float, default=0.99)
    p.add_argument("--lgcp_beta", type=float, default=0.01)
    p.add_argument("--learn_inducing", action="store_true", default=False)

    p.add_argument("--grid_padding", type=float, default=0.02)
    p.add_argument("--grid_oversample", type=float, default=3.0)
    p.add_argument("--min_sep", type=float, default=1e-3)
    p.add_argument("--kmeans_batch_size", type=int, default=4096)
    p.add_argument("--min_sep_passes", type=int, default=3)

    p.add_argument("--gp_iters", type=int, default=800)
    p.add_argument("--gp_lr", type=float, default=0.01)
    p.add_argument("--lr_schedule", type=str, choices=["none", "cosine", "cosine_wr"], default="none", help="Learning rate scheduler for GP training.")
    p.add_argument("--mean_cols", type=int, nargs="+", required=True, help="Column indices used in the LinearMean (e.g. --mean_cols 2 3).")
    p.add_argument("--ols_init_mean", action="store_true", default=False)
    p.add_argument("--latent_col", type=int, default=4)
    # p.add_argument("--latent_combine", type=str, choices=["add", "product", "sum_product", "cov_plus_mult_latent_plus_coords", "latent_local_and_global"])

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--torch_dtype", type=str, choices=["float32", "float64"], default="float32")

    p.add_argument("--rescale_inputs", action="store_true", default=False)
    p.add_argument("--xy_rescale_factor", type=float, default=500.)
    p.add_argument("--elev_rescale_factor", type=float, default=10.)
    p.add_argument("--ts_rescale_factor", type=float, default=1.)

    p.add_argument("--noaa_data_type", type=str, choices=["trav", "raster", "trav_raster"], default="trav", help="Type of NOAA data to use for validation: 'trav' for traversal-based, 'raster' for raster-aggregate.")
    p.add_argument("--run_residual_diagnostic", action="store_true", default=False)
    p.add_argument("--hide_kernel_summary", action="store_true", default=False)
    p.add_argument("--hide_validation_corrs", action="store_true", default=False)
    p.add_argument("--make_plots", action="store_true", default=False)
    p.add_argument("--decompose_gp", action="store_true", default=False, help="Whether to decompose the GP posterior into components (only if latent_in_cov is True).")

    p.add_argument("--k_trials", type=int, default=1,
               help="Number of random trials (different sensor subsamples) per n_sensors.")
    p.add_argument("--save_trial_maps", action="store_true", default=False,
                help="If set, saves maps/plots for every trial (otherwise only trial 0).")


    return p.parse_args()


def main() -> None:

    args = parse_args()

    # create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    if args.city_name.lower() != "chapel_hill" and not args.noaa_dir:
        raise ValueError("--noaa_dir is required unless city_name=chapel_hill")

    # get raw sensor data
    sensor_csv = os.path.join(args.data_root, "data", args.wu_date, "sensor_data", f"wu_{args.noaa_date}_all_day_nc.csv")
    assert os.path.exists(sensor_csv), f"Missing sensor CSV: {sensor_csv}"
    df = pd.read_csv(sensor_csv)

    df_slot = filter_data_to_time_slot(
        df=df,
        time_slot=args.time_slot,
        metric_col=args.metric_col,
        last10=args.last10,
    )
    station_summary = get_median_per_station(
        df_slot=df_slot,
        metric_col=args.metric_col
    )

    tot_sensors_unique = station_summary["station_id"].nunique()
    print(f"Total number of unique sensors in {args.time_slot}: {tot_sensors_unique}")
    start = min(500, tot_sensors_unique)
    sensor_grid = np.unique(np.linspace(start, tot_sensors_unique, num=5, dtype=int))
    print("Num_sensors grid:", sensor_grid)

    # now we'd like to write a for-loop over sensor_grid, and for each n_sensors, 
    # run_lgcp_exactgp_experiment on the subsampled data stored in the csv made during get_randomized_areal_data

    # 1. get_randomized_areal_data for all n_sensors in sensor_grid
    # 2. run_lgcp_exactgp_experiment on each csv
    # 3. validate against noaa traversal/raster data and get correlation

    # step 1
    results: list[Dict[str, Any]] = []
    total_runs = len(sensor_grid) * args.k_trials
    run_idx = 0

    for n_sensors in map(int, sensor_grid):
        print(f"Running experiment with n_sensors={n_sensors}...")

        # the output dir should be different for each n_sensors setting
        n_dir = os.path.join(args.output_dir, f"n_sensors_{n_sensors}")
        os.makedirs(n_dir, exist_ok=True)

        for k0 in range(args.k_trials):
            trial_id = k0 + 1 # 1-based trial index for display
            run_idx += 1
            print(f"Running n_sensors={n_sensors}, trial={trial_id}/{args.k_trials} ({run_idx}/{total_runs})")

            # per-trial directory: .../n_sensors_{n}/trial_{t}/
            trial_dir = os.path.join(n_dir, f"trial_{trial_id}")
            os.makedirs(trial_dir, exist_ok=True)
            # deterministic seed for this (n, trial)
            trial_seed = make_trial_seed(args.seed, n_sensors, k0, args.k_trials)

            # new idea: create an experiment output directory   
            exp_out_dir = os.path.join(trial_dir, "exp")
            os.makedirs(exp_out_dir, exist_ok=True)

            save_maps = args.save_trial_maps or (trial_id == 1)

            subsampled_csv_path = get_randomized_areal_data(
                station_summary=station_summary,
                n_keep=n_sensors,
                wu_date=args.wu_date,
                noaa_date=args.noaa_date,
                time_slot=args.time_slot,
                data_root=args.data_root,
                out_dir=trial_dir,  # <-- IMPORTANT: send outputs to per-trial folder
                city_name=args.city_name,
                census_group_type=args.census_group_type,
                trial=trial_id,     # <-- pass 1-based for filenames
                trial_seed=trial_seed,
                save_maps=save_maps,
            )

            # ideally we'd only want run_residual_diagnostic, decompose_gp to happen for trial 0
            run_residual_diagnostic = args.run_residual_diagnostic and (trial_id == 1)
            decompose_gp = args.decompose_gp and (trial_id == 1)

            # step 2
            result_row = run_lgcp_exactgp_experiment(
                # paths / identifiers
                data_dir=args.data_root,
                geo_csv_override=subsampled_csv_path, # your kept-sensor CSV
                noaa_dir=args.noaa_dir,
                output_dir=exp_out_dir, # <-- IMPORTANT: send outputs to per-trial folder
                wu_date=args.wu_date,
                census_group_type=args.census_group_type,
                city_name=args.city_name,
                time_slot=args.time_slot,

                # LGCP knobs
                m_inducing=args.m_inducing,
                inducing_type=args.inducing_type,
                lgcp_iters=args.lgcp_iters,
                lgcp_lr=args.lgcp_lr,
                lgcp_particles=args.lgcp_particles,
                lgcp_lrd=args.lgcp_lrd,
                lgcp_beta=args.lgcp_beta,
                learn_inducing=args.learn_inducing,

                grid_padding=args.grid_padding,
                grid_oversample=args.grid_oversample,
                min_sep=args.min_sep,
                kmeans_batch_size=args.kmeans_batch_size,
                min_sep_passes=args.min_sep_passes,

                # ExactGP knobs
                latent_in_cov=True, # only interested in LGCP+GP with latent in cov here (Linear Kernel)
                latent_col=args.latent_col,
                latent_kernel="linear",
                latent_ls_config=None, # should be None for linear kernel
                latent_combine="sum_product",

                gp_iters=args.gp_iters,
                gp_lr=args.gp_lr,
                lr_schedule=args.lr_schedule,
                mean_cols=args.mean_cols,
                ols_init_mean=args.ols_init_mean,

                # dtype / scaling
                seed=trial_seed,   # <-- IMPORTANT: vary per trial (or use args.seed if you want fixed training randomness)
                torch_dtype=args.torch_dtype,
                rescale_inputs=args.rescale_inputs,
                xy_rescale_factor=args.xy_rescale_factor,
                elev_rescale_factor=args.elev_rescale_factor,
                ts_rescale_factor=args.ts_rescale_factor,
                
                # misc / plotting stuff
                noaa_data_type=args.noaa_data_type,
                hide_validation_corrs=args.hide_validation_corrs,
                hide_kernel_summary=args.hide_kernel_summary,
                run_residual_diagnostic=run_residual_diagnostic,
                make_plots=args.make_plots,
                decompose_gp=decompose_gp,
            )

            # add experiment name to result_row
            result_row["exp_name"] = (
                f"lgcp_gp_sensor_sweep__{args.wu_date}__{args.city_name}__{args.time_slot}__"
                f"{args.census_group_type}__n{n_sensors}__trial{trial_id}"
            )
            result_row["n_sensors"] = n_sensors
            result_row["trial"] = trial_id
            result_row["trial_seed"] = trial_seed
            result_row["trial_dir"] = trial_dir

            results.append(result_row)

            # append to output CSV
            row_path = os.path.join(trial_dir, "result_row.csv")
            pd.DataFrame([result_row]).to_csv(row_path, index=False)
            # output_csv_path = os.path.join(
            #     args.output_dir,
            #     f"{args.wu_date}_{args.city_name}_{args.time_slot}_sensor_sweep_results.csv"
            # )
            # file_exists = os.path.exists(output_csv_path)
            # output_df = pd.DataFrame([result_row])
            # output_df.to_csv(output_csv_path, mode="a", header=not file_exists, index=False)
            print_single_result(run_idx, total_runs, result_row)

            print(f"[DONE] n_sensors={n_sensors}, trial={trial_id}\n")

    # after all experiments are done, make the final plot
    plot_corr_vs_n_sensors(
        results,
        out_dir=args.output_dir,
        wu_date=args.wu_date,
        city_name=args.city_name,
        time_slot=args.time_slot,
        census_group_type=args.census_group_type,
        noaa_data_type=args.noaa_data_type,
        exp_name_hint="lgcp_gp_sensor_sweep",
    )

    # and save everything to one big csvL
    final_path = os.path.join(
        args.output_dir,
        f"{args.wu_date}_{args.city_name}_{args.time_slot}_sensor_sweep_results.csv"
    )
    final_df = pd.DataFrame(results).sort_values(["n_sensors", "trial"])
    final_df.to_csv(final_path, index=False)
    print(f"[CSV] Wrote final results to {final_path}")



if __name__ == "__main__":
    main()





