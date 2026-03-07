# src/data/nc_areal_dataset.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


@dataclass
class DataBundle:
    df_geo: pd.DataFrame
    df_elev: pd.DataFrame
    df_full: pd.DataFrame
    df_obs: pd.DataFrame
    areal_units_all: gpd.GeoDataFrame
    water_ids: set
    imputer: KNNImputer
    t_s_col: str
    temp_col: str


@dataclass
class GPFeatureBuilder:
    """
    Builds features for Gaussian Process (GP) modeling.
    Rescales features based on input factors: xy_unit, elev_unit, ts_unit.
    """
    mu_xy: np.ndarray
    xy_denom: float
    mu_elev: float
    elev_denom: float
    mu_ts: float
    ts_denom: float
    latent_in_cov: bool
    scaler_latent: Optional[StandardScaler]

    @classmethod
    def build_train(cls, df_obs, t_s_col: str, latent_train: Optional[np.ndarray] = None, *, mean_cols: list[int], latent_col: int, latent_in_cov: bool = True,
        xy_unit: float = 500.0, elev_unit: float = 10.0, ts_unit: float = 1.0, rescale_inputs: bool = False,
    ) -> tuple["GPFeatureBuilder", np.ndarray]:
        """
        Builds X_train with columns:
        [xy_scaled, elev_scaled, ts_scaled] (+ latent_scaled if latent_in_cov=True or if latent_col in mean_cols)
        """
        # xy in meters
        g_obs = gpd.GeoDataFrame(df_obs.copy(), geometry=gpd.points_from_xy(df_obs["lon"], df_obs["lat"]), crs="EPSG:4326",).to_crs("EPSG:32119")
        xy_train = np.column_stack([g_obs.geometry.x.to_numpy(), g_obs.geometry.y.to_numpy()])
        mu_xy = xy_train.mean(axis=0); sd_xy = xy_train.std(0)
        xy_denom = sd_xy if (rescale_inputs is False) else float(xy_unit)
        xy_scaled = (xy_train - mu_xy) / xy_denom

        elev = df_obs["elevation"].values.copy()
        ts = df_obs[t_s_col].values.copy()
        mu_elev, mu_ts = float(elev.mean()), float(ts.mean())
        sd_elev, sd_ts = float(elev.std()), float(ts.std())

        elev_denom = sd_elev if (rescale_inputs is False) else float(elev_unit)
        ts_denom = sd_ts if (rescale_inputs is False) else float(ts_unit)
        elev_scaled = (elev - mu_elev) / elev_denom
        ts_scaled = (ts - mu_ts) / ts_denom

        X_parts = [xy_scaled, elev_scaled, ts_scaled]

        scaler_latent = None
        if latent_in_cov or (mean_cols is not None and any(col == latent_col for col in mean_cols)):
            if latent_train is None:
                raise ValueError("latent_in_cov=True but latent_train is None")

            latent = np.asarray(latent_train, dtype=float).reshape(-1, 1)
            scaler_latent = StandardScaler()
            latent_scaled = scaler_latent.fit_transform(latent).ravel()
            X_parts.append(latent_scaled)

        X_train = np.column_stack(X_parts)

        builder = cls(
            mu_xy=mu_xy,
            xy_denom=xy_denom,
            mu_elev=mu_elev,
            elev_denom=elev_denom,
            mu_ts=mu_ts,
            ts_denom=ts_denom,
            latent_in_cov=latent_in_cov,
            scaler_latent=scaler_latent,
        )

        return builder, X_train
    
    def build_test(self, *, areal_units_all, df_elev, modis_df, t_s_col: str, latent_col, mean_cols, imputer, pred_field_latent=None):
        if {"centr_lat", "centr_lon"}.issubset(areal_units_all.columns):
            df_full_all = (
                areal_units_all[["GEOID", "centr_lat", "centr_lon"]]
                .merge(df_elev[["GEOID", "elevation"]].drop_duplicates("GEOID"), on="GEOID", how="left")
                .merge(modis_df[["GEOID", t_s_col]].drop_duplicates("GEOID"), on="GEOID", how="left")
            ).rename(columns={"centr_lat": "lat", "centr_lon": "lon"})
        else:
            raise ValueError("areal_units_all must contain 'centr_lat' and 'centr_lon' columns")
        df_full_all["GEOID"] = df_full_all["GEOID"].astype(str)
        df_full_all[["elevation", t_s_col]] = imputer.transform(df_full_all[["elevation", t_s_col]])

        g_all_proj = areal_units_all.to_crs("EPSG:32119")
        centroids = g_all_proj.geometry.centroid
        xy_test = np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])

        xy_test_s = (xy_test - self.mu_xy) / self.xy_denom
        elev_test = df_full_all["elevation"].to_numpy(dtype=float)
        ts_test = df_full_all[t_s_col].to_numpy(dtype=float)
        elev_test_s = (elev_test - self.mu_elev) / self.elev_denom
        ts_test_s = (ts_test - self.mu_ts) / self.ts_denom
        X_parts = [xy_test_s, elev_test_s, ts_test_s]

        if self.latent_in_cov or (mean_cols is not None and latent_col in mean_cols):
            if pred_field_latent is None:
                raise ValueError("pred_field_latent is required when latent_in_cov=True")
            if self.scaler_latent is None:
                raise RuntimeError("latent_in_cov=True but scaler_latent is None")
            latent_test_s = self.scaler_latent.transform(np.asarray(pred_field_latent, dtype=float).reshape(-1, 1))
            X_parts.append(latent_test_s)

        assert (df_full_all["GEOID"].to_numpy() == areal_units_all["GEOID"].astype(str).to_numpy()).all()
        test_raw = np.column_stack(X_parts)
        return df_full_all, test_raw


from pathlib import Path


class NCArealDataset:
    """
    Prep areal data from Weather Underground (aggregated to tracts or block groups) before GP modeling.

    Responsibilities:
    - load arealized csv/shp inputs (one row per GEOID)
    - remove water-only units consistently
    - compute centroids (lat/lon)
    - merge covariates (elev + MODIS t_s)
    - KNN-impute features using [lat, lon, elevation, t_s]
    - provide df_full / df_obs + areal_units_all (non-water geometry)
    - build standardized GP design matrices (train + test)
    """

    def __init__(self, data_dir: str, wu_date: str, census_group_type: str, time_slot: str, imputer_neighbors: int = 5, imputer_weights: str = "distance",
    ):
        self.data_dir = data_dir
        self.wu_date = wu_date
        self.census_group_type = census_group_type
        self.time_slot = time_slot
        self.t_s_col = f"t_s_{time_slot}"
        self.temp_col = f"{time_slot}_temp"
        self.imputer_neighbors = imputer_neighbors
        self.imputer_weights = imputer_weights
        self.bundle: Optional[DataBundle] = None

    def _paths(self) -> Tuple[str, str, str]:
        t_s_col, time_slot, data_dir, wu_date = self.t_s_col, self.time_slot, self.data_dir, self.wu_date

        if self.census_group_type == "tract":
            geo_csv = os.path.join(data_dir, "data", wu_date, f"nc_tracts_temp_pop_{time_slot}.csv")
            elev_csv = os.path.join(data_dir, "data", wu_date, f"nc_tracts_elev_{t_s_col}_{time_slot}.csv")
            shp_file = os.path.join(data_dir, "data", wu_date, f"nc_tracts_temp_pop_{time_slot}.shp")
        elif self.census_group_type == "block_group":
            geo_csv = os.path.join(data_dir, "data", wu_date, f"nc_bg_temp_pop_{time_slot}.csv")
            elev_csv = os.path.join(data_dir, "data", wu_date, f"nc_bg_elev_{t_s_col}_{time_slot}.csv")
            shp_file = os.path.join(data_dir, "data", wu_date, f"nc_bg_temp_pop_{time_slot}.shp")
        else:
            raise ValueError("Invalid census_group_type. Choose 'tract' or 'block_group'.")

        return geo_csv, elev_csv, shp_file

    def _load_inputs(self) -> Tuple[pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame]:
        geo_csv, elev_csv, shp_file = self._paths()
        df_geo, df_elev, areal_units = pd.read_csv(geo_csv), pd.read_csv(elev_csv), gpd.read_file(shp_file)
        
        for df in (df_geo, df_elev):
            df["GEOID"] = df["GEOID"].astype(str)
        areal_units["GEOID"] = areal_units["GEOID"].astype(str)

        return df_geo, df_elev, areal_units

    @staticmethod
    def _remove_water_units(areal_units: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, set]:
        water_ids = set(areal_units.loc[(areal_units["ALAND"] == 0) & (areal_units["AWATER"] > 0), "GEOID"])
        out = areal_units.loc[~areal_units["GEOID"].isin(water_ids)].copy()
        return out, water_ids

    @staticmethod
    def _add_centroids_latlon(areal_units: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        geo_proj = areal_units.to_crs("EPSG:32119").copy()
        geo_proj["centroid"] = geo_proj.geometry.centroid
        centroids_latlon = geo_proj.set_geometry("centroid").to_crs("EPSG:4326")
        geo_proj["lon"] = centroids_latlon.geometry.x
        geo_proj["lat"] = centroids_latlon.geometry.y
        return geo_proj

    def _build_df_full(self, df_geo: pd.DataFrame, df_elev: pd.DataFrame, areal_units_nonwater: gpd.GeoDataFrame,
    ) -> pd.DataFrame:

        df_full = pd.merge(df_geo, df_elev[["GEOID", "elevation"]], on="GEOID", how="left")
        geo_proj = self._add_centroids_latlon(areal_units_nonwater)
        df_full = df_full.merge(geo_proj[["GEOID", "lat", "lon"]].copy(), on="GEOID", how="left")
        modis_df = df_elev[["GEOID", self.t_s_col]].copy()
        df_full = df_full.merge(modis_df, on="GEOID", how="left")

        return df_full

    def _impute_features(self, df_full: pd.DataFrame) -> Tuple[pd.DataFrame, KNNImputer]:
        t_s_col = self.t_s_col
        impute_cols = ["elevation", t_s_col]
        df_for_impute = df_full[impute_cols].copy()

        imputer = KNNImputer(n_neighbors=self.imputer_neighbors, weights=self.imputer_weights)
        imputed = imputer.fit_transform(df_for_impute)

        df_full = df_full.copy()
        df_full["elevation"] = imputed[:, 0]
        df_full[t_s_col] = imputed[:, 1]
        return df_full, imputer

    def prepare_data(self) -> DataBundle:
        df_geo, df_elev, areal_units = self._load_inputs()

        areal_units_nonwater, water_ids = self._remove_water_units(areal_units)
        df_geo = df_geo.loc[~df_geo["GEOID"].isin(water_ids)].copy()
        df_elev = df_elev.loc[~df_elev["GEOID"].isin(water_ids)].copy()

        df_full = self._build_df_full(df_geo, df_elev, areal_units_nonwater)
        df_full, imputer = self._impute_features(df_full)

        mask = ~np.isnan(df_full[self.temp_col].values)
        df_obs = df_full.loc[mask].copy()

        areal_units_all = areal_units_nonwater.copy()
        geo_proj = self._add_centroids_latlon(areal_units_all)

        areal_units_all["centr_lon"] = geo_proj["lon"].to_numpy()
        areal_units_all["centr_lat"] = geo_proj["lat"].to_numpy()


        self.bundle = DataBundle(
            df_geo=df_geo,
            df_elev=df_elev,
            df_full=df_full,
            df_obs=df_obs,
            areal_units_all=areal_units_all,
            water_ids=water_ids,
            imputer=imputer,
            t_s_col=self.t_s_col,
            temp_col=self.temp_col,
        )
        return self.bundle