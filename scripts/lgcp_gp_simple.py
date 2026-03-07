"""
LGCP + ExactGP training pipeline script.

Author: Ellie Kim
"""

from typing import List, Optional, Dict, Any
import time
import torch
import gpytorch
import matplotlib.pyplot as plt
import os

from nctempps.trainers.train import train_pyro_model
from utils import (set_seeds)
from nctempps.datasets.nc_areal_dataset import (NCArealDataset, GPFeatureBuilder)
from nctempps.trainers.train import train_exact_gp
from nctempps.models.svlgcp import build_lgcp, attach_lgcp_outputs


@torch.no_grad()
def mean_at(model, X):
    idx = X.new_tensor(model.mean_cols, dtype=torch.long)
    return model.mean_module(X.index_select(dim=-1, index=idx))

def kernel_signature(k) -> str:
    """Returns a readable string like 'spatial' or '(spatial*latent)'."""
    # unwrap ScaleKernel for labeling
    if isinstance(k, gpytorch.kernels.ScaleKernel):
        return kernel_signature(k.base_kernel)

    # AdditiveKernel: sum of terms
    if isinstance(k, gpytorch.kernels.AdditiveKernel):
        return " + ".join(kernel_signature(kk) for kk in k.kernels)

    # ProductKernel: product of factors
    if isinstance(k, gpytorch.kernels.ProductKernel):
        return "*".join(kernel_signature(kk) for kk in k.kernels)

    # Your specific bases:
    if isinstance(k, gpytorch.kernels.MaternKernel):
        ad = list(map(int, k.active_dims)) if getattr(k, "active_dims", None) is not None else None
        if ad == [0, 1]:
            return "spatial"
        if ad == [2, 3]:
            return "cov"   # elev+t_s term in your cov_plus_mult_latent_plus_coords
        return f"matern{ad}"

    if isinstance(k, gpytorch.kernels.RBFKernel):
        ad = list(map(int, k.active_dims)) if getattr(k, "active_dims", None) is not None else None
        return f"rbf{ad}"

    # generic fallback
    return type(k).__name__


def _parse_wu_date_mmddyy(wu_date: str) -> tuple[int, int, int]:
    """
    Parse 'MMDDYY' (e.g., '071321') -> (year, month, day) as ints.
    """
    wu_date = str(wu_date)
    if len(wu_date) != 6 or not wu_date.isdigit():
        raise ValueError(f"wu_date must be 'MMDDYY' (e.g. '071321'), got {wu_date!r}")
    month = int(wu_date[0:2])
    day   = int(wu_date[2:4])
    year  = 2000 + int(wu_date[4:6])
    return year, month, day

from nctempps.datasets.process_station_data import process_one_day_slot

def run_lgcp_exactgp_experiment(
    *,
    data_dir: str,
    wu_station_csv: str,
    cov_root: str,
    output_dir: str,
    wu_date: str,
    census_group_type: str,
    city_name: str,
    time_slot: str,
    m_inducing: int,
    inducing_type: str,
    lgcp_iters: int,
    lgcp_lr: float,
    lgcp_particles: int,
    lgcp_lrd: float,
    lgcp_beta: float,
    learn_inducing: bool,
    grid_padding: float,
    grid_oversample: float,
    min_sep: float,
    kmeans_batch_size: int,
    min_sep_passes: int,
    latent_in_cov: bool,
    latent_col: int,
    latent_kernel: Optional[str],
    latent_ls_config: Optional[Dict[str, Any]],
    latent_combine: str,
    gp_iters: int,
    gp_lr: float,
    lr_schedule: str,
    mean_cols: List[int],
    ols_init_mean: bool,
    seed: int,
    torch_dtype: str,
    make_plots: bool,
    save_pred_csv: Optional[str] = None,
    pred_col_name: str = "pred",
):
    """
    Runs LGCP + ExactGP pipeline and returns metrics + predictions.
    """
    print("[CONFIG] wu date: ", wu_date)
    print("[CONFIG] census group type: ", census_group_type)
    dtype = set_seeds(seed, torch_dtype)
    print("[CONFIG] mean_cols:", mean_cols)
    print("[CONFIG] ols_init_mean:", ols_init_mean)
    print("[CONFIG] latent_in_cov:", latent_in_cov)
    print("[CONFIG] latent_kernel:", latent_kernel)
    print("[CONFIG] latent_ls_config:", latent_ls_config)
    print("[CONFIG] kernel combination method:", latent_combine)
    print("[CONFIG] lr_schedule:", lr_schedule)
    print("[CONFIG] make_plots:", make_plots)
    print("[CONFIG] save_pred_csv:", save_pred_csv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device} cuda_available={torch.cuda.is_available()}", flush=True)
    if device.type == "cuda":
        print(f"[DEVICE] gpu_name={torch.cuda.get_device_name(0)}", flush=True)

    # load WU station data and arealize it

    # determine which WU point-wise data file to use
    wu_csv = wu_station_csv  # example: /hpc/group/carlsonlab/weather_underground/july/071321/4pm.csv

    # enforce new convention
    time_slot = time_slot.lower()
    if time_slot not in {"6am", "4pm"}:
        raise ValueError(f"time_slot must be '6am' or '4pm', got {time_slot!r}")
    
    year, month, day = _parse_wu_date_mmddyy(wu_date)

    # 2) arealize + write files in the exact naming convention NCArealDataset expects
    outs = process_one_day_slot(
        wu_csv=wu_station_csv,                  # e.g. /.../071321/4pm.csv
        time_slot=time_slot,                    # "6am" or "4pm"
        year=year,
        month=month,
        cov_root=cov_root,
        out_root=data_dir, 
        wu_date_folder=wu_date,                   # e.g. "071321"
    )

    # 3) now NCArealDataset can read those freshly-created files
    dataset = NCArealDataset(
        data_dir=data_dir,
        wu_date=wu_date,
        census_group_type=census_group_type,
        time_slot=time_slot,
    )
    data = dataset.prepare_data()

    for col in ("population", "count"):
        if col not in data.df_geo.columns:
            raise ValueError(f"df_geo missing '{col}'")

    # # ensure gdf has pop/count
    # areal_units_all = data.areal_units_all.merge(
    #     data.df_geo[["GEOID", "population", "count"]],
    #     on="GEOID",
    #     how="left",
    # )
    #########################################################################################
    
    t_s_col = data.t_s_col
    temp_col = data.temp_col
    for col in ("population", "count"):
        if col not in data.areal_units_all.columns:
            raise ValueError(
                f"areal_units_all shapefile is missing required column '{col}'. "
                f"Make sure {census_group_type} shp has population + count."
            )

    # LGCP setup + train
    lgcp_model, _, test_X_lgcp, train_X_lgcp, _, _ = build_lgcp(areal_units_all=data.areal_units_all, df_obs=data.df_obs, m=m_inducing, seed=seed, dtype=dtype,
                      inducing_type=inducing_type, grid_padding=grid_padding, grid_oversample=grid_oversample,
                      min_sep=min_sep, kmeans_batch_size=kmeans_batch_size, max_passes=min_sep_passes, beta=lgcp_beta,
                      learn_inducing=learn_inducing)
    # population = torch.from_numpy(data.areal_units_all["population"].to_numpy() / 1000.0 + 1e-2).to(dtype=torch.float32)
    # counts = torch.from_numpy(data.areal_units_all["count"].to_numpy()).to(dtype=torch.float32)

    # switch to GPU:
    lgcp_model = lgcp_model.to(device)
    test_X_lgcp = test_X_lgcp.to(device)
    train_X_lgcp = train_X_lgcp.to(device)

    population = torch.from_numpy(data.areal_units_all["population"].to_numpy() / 1000.0 + 1e-2)\
        .to(device=device, dtype=torch.float32)
    counts = torch.from_numpy(data.areal_units_all["count"].to_numpy())\
        .to(device=device, dtype=torch.float32)

    print("Training LGCP...")
    lgcp_model.train()
    # add print statements to make sure it's doing this on GPU
    print("[CHECK] lgcp_model device:", next(lgcp_model.parameters()).device, flush=True)
    print("[CHECK] test_X_lgcp device:", test_X_lgcp.device, "counts:", counts.device, "population:", population.device, flush=True)

    _ = train_pyro_model(model=lgcp_model, X=test_X_lgcp, counts=counts, population=population,
        lr=lgcp_lr, num_iter=lgcp_iters, num_particles=lgcp_particles, lrd=lgcp_lrd)
    
    lgcp_model.eval()
    # attach LGCP posterior latent means to areal units (under col "latent_mean")
    areal_units_all = attach_lgcp_outputs(areal_units_all=data.areal_units_all, lgcp_model=lgcp_model, test_X_lgcp=test_X_lgcp)

    # add an extra check just to be sure the temp_col we want to predict is actually present in the dataframes we have
    temp_col = data.temp_col
    if temp_col not in data.areal_units_all.columns:
        raise ValueError(
            f"areal_units_all is missing required temperature column '{temp_col}'. "
            f"Columns present: {list(data.areal_units_all.columns)}"
        )

    # build latent_train for training ExactGP
    with torch.no_grad():
        latent_train_t = lgcp_model(train_X_lgcp).mean
    latent_train = latent_train_t.detach().cpu().numpy().reshape(-1)

    # build rest of training data for exactGP
    builder, X_train_np = GPFeatureBuilder.build_train(df_obs=data.df_obs, t_s_col=t_s_col, latent_train=latent_train, mean_cols=mean_cols, latent_col=latent_col, latent_in_cov=latent_in_cov,
        xy_unit=None, elev_unit=None, ts_unit=None, rescale_inputs=False)
    y_train = data.df_obs[temp_col].to_numpy()
    train_X_gp = torch.as_tensor(X_train_np, device=device, dtype=dtype)
    # train_y_gp = torch.from_numpy(y_train).to(device=device, dtype=dtype)
    train_y_gp = torch.as_tensor(y_train, device=device, dtype=dtype)

    # start timer to see how long it takes to train
    print("Training ExactGP...")
    t0 = time.perf_counter()
    # ---- FORCE ExactGP latent kernel config ----
    latent_in_cov = True
    latent_kernel = "linear"
    latent_ls_config = {"mode": "none"}   # no lengthscale constraint / config
    latent_combine = "sum_product"  # K  =  σ1 * K_spatial + σ2 * (K_spatial2 *  Linear(latent))
    exactgp_model, likelihood, losses = train_exact_gp(
        output_dir=output_dir,
        train_X=train_X_gp,
        train_y=train_y_gp,
        lr=gp_lr,
        num_iter=gp_iters,
        mean_cols=mean_cols,
        ols_init_mean=ols_init_mean,
        latent_in_cov=latent_in_cov,
        latent_col=latent_col,
        latent_kernel=latent_kernel,
        latent_ls_config=latent_ls_config,
        latent_combine=latent_combine,
        lr_schedule=lr_schedule, # options: none, cosine, cosine_wr (with warm restarts)
        device=device,          # <-- add
        dtype=dtype,            # <-- optional but recommended
    )
    gp_seconds = time.perf_counter() - t0
    gp_loss_last = float(losses[-1]) if losses else float("nan")

    # add print statements to make sure it's doing this on GPU
    print("[CHECK] exactgp_model param device:", next(exactgp_model.parameters()).device, flush=True)
    print("[CHECK] likelihood param device:", next(likelihood.parameters()).device, flush=True)
    print("[CHECK] train_X device:", train_X_gp.device, "train_y device:", train_y_gp.device, flush=True)

    # get predicted latent at all centroids (in standardized meters space) from LGCP
    with torch.no_grad():
        pred_field_latent = lgcp_model(test_X_lgcp).mean.detach().cpu().numpy()

    # build test set features + predict with ExactGP
    modis_df = data.df_elev[["GEOID", t_s_col]].copy()
    _, test_X_gp = builder.build_test(areal_units_all=areal_units_all, df_elev=data.df_elev, modis_df=modis_df, t_s_col=t_s_col, mean_cols=mean_cols, latent_col=latent_col,
        imputer=data.imputer, pred_field_latent=pred_field_latent,   # only required if latent_in_cov=True
    )
    test_X_gp = torch.as_tensor(test_X_gp, device=device, dtype=dtype)

    # predict with your trained exactgp and attach predictions to the gdf
    exactgp_model.eval()
    likelihood.eval()
    with torch.no_grad():
        observed_pred = likelihood(exactgp_model(test_X_gp))
        mean_ps = observed_pred.mean.detach().cpu().numpy()

    tracts_pred = areal_units_all.copy()
    tracts_pred["GEOID"] = tracts_pred["GEOID"].astype(str)
    tracts_pred["exact_gp"] = mean_ps

    # Save pred_csv to location specified:
    if save_pred_csv is not None:
        os.makedirs(os.path.dirname(save_pred_csv), exist_ok=True)
        out_df = tracts_pred[["GEOID", "exact_gp"]].rename(columns={"exact_gp": pred_col_name})
        out_df.to_csv(save_pred_csv, index=False)
        print(f"[WRITE] saved predictions -> {save_pred_csv}")
    else:
        print("[WRITE] save_pred_csv not specified, skipping saving predictions to csv.")

    # Also plot histogram of predictions (save figure as png)
    # NOTE: This is the only diagnostic we'll do -- we'll just check to make sure the predictions are within a similar range to the observations
    if make_plots:
        plt.figure(figsize=(8, 6))
        plt.hist(tracts_pred[f'{time_slot}_temp'], label='obs', alpha=0.5)
        plt.hist(tracts_pred['exact_gp'], label='pred', alpha=0.5)
        plt.legend()
        hist_fig_path = os.path.join(output_dir, f"{wu_date}_nc_{census_group_type}_{time_slot}_obs_vs_pred_hist.png".replace("/", "_"))
        plt.savefig(hist_fig_path, dpi=200, bbox_inches="tight")
        plt.close()

    result = {
        # identifiers
        "wu_date": wu_date,
        "time_slot": time_slot,
        "census_group_type": census_group_type,
        "city_name": city_name,
        "m_inducing": int(m_inducing),
        "inducing_type": inducing_type,

        # config
        "latent_in_cov": bool(latent_in_cov),
        "latent_kernel": latent_kernel if latent_in_cov else None,
        "latent_ls_config": latent_ls_config if latent_in_cov else None,

        "gp_train_seconds": float(gp_seconds),
        "gp_loss_last": float(gp_loss_last),
    }

    return result