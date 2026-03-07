"""
Sparse variational LGCP model definitions

"""

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from typing import Tuple
import torch
import pyro
import pyro.distributions as dist
import gpytorch
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel, ProductKernel, AdditiveKernel
from gpytorch.priors import GammaPrior
from gpytorch.constraints import Interval
from sklearn.cluster import MiniBatchKMeans
from typing import Literal, Optional
from shapely.geometry import Point
from shapely.prepared import prep
from utils import geoseries_union_all


def select_inducing_with_min_sep(points_np, m=1000, min_sep=1e-3, seed=42, max_passes=3):
    """
    Greedy selection: random order, keep a point only if it is at least `min_sep`
    away (Euclidean distance) from all previously kept points.
    If we can't reach m, we progressively relax `min_sep` by half up to max_passes.

    NB: `min_sep` is in the same units as `points_np` (e.g., standardized meters).
    """
    rng = np.random.default_rng(seed)
    points_np = points_np[np.isfinite(points_np).all(axis=1)]  # drop NaN/Inf
    points_np = np.unique(points_np, axis=0)  # drop exact dups

    for _ in range(max_passes):
        keep = []
        perm = rng.permutation(len(points_np))
        min_sep2 = float(min_sep) ** 2

        for idx in perm:
            p = points_np[idx]
            if not keep:
                keep.append(p)
            else:
                # squared distance to all kept
                d2 = np.sum((np.asarray(keep) - p) ** 2, axis=1)
                if np.min(d2) > min_sep2:
                    keep.append(p)
            if len(keep) >= m:
                break

        if len(keep) >= m:
            keep = np.asarray(keep)[:m]
            break
        else:
            min_sep *= 0.5 # relax the threshold and try again

    keep = np.asarray(keep)
    # If still short, randomly top up without replacement (duplicates already removed)
    if keep.shape[0] < m:
        remaining = points_np[
            ~np.in1d(
                points_np.view([("", points_np.dtype)] * 2),
                keep.view([("", keep.dtype)] * 2),
            )
        ]
        need = min(m - keep.shape[0], remaining.shape[0])
        if need > 0:
            keep = np.vstack(
                [keep, remaining[rng.choice(len(remaining), size=need, replace=False)]]
            )
    
    keep = keep + rng.standard_normal(keep.shape) * 1e-9 # tiny jitter to avoid exact equalities
    return keep[:m]


InducingType = Literal["grid", "kmeans", "random", "min_separation"]

def initialize_inducing_points(
    *, xy_all_m: np.ndarray, m: int, inducing_type: InducingType = "kmeans", seed: int = 0,
    mask_geom_m=None, grid_padding: float = 0.02, grid_oversample: float = 3.0,
    kmeans_batch_size: int = 4096, min_sep: Optional[float] = 1e-3, max_passes: int = 3,
) -> np.ndarray:
    """
    Returns inducing points Z in STANDARDIZED space, consistent with:
        xy_all_s = (xy_all_m - mu) / sd

    NB: `random` selects m points from xy_all_m randomly.
    """
    # input validation
    xy_all_m = np.asarray(xy_all_m, dtype=float)
    if xy_all_m.ndim != 2 or xy_all_m.shape[1] != 2:
        raise ValueError(f"xy_all_m must be (N,2); got {xy_all_m.shape}")
    if m <= 0:
        raise ValueError("m must be positive")

    rng = np.random.default_rng(seed)

    # standardize xy coords before selecting inducing points
    mu = xy_all_m.mean(axis=0); sd = xy_all_m.std(axis=0) + 1e-12
    xy_all_s = (xy_all_m - mu) / sd

    # bbox in standardized space
    mins_s, maxs_s = xy_all_s.min(axis=0), xy_all_s.max(axis=0)
    span_s = np.maximum(maxs_s - mins_s, 1e-12)

    if inducing_type == "kmeans":
        km = MiniBatchKMeans(n_clusters=m, batch_size=kmeans_batch_size, random_state=seed, n_init=3)
        km.fit(xy_all_s)
        return km.cluster_centers_

    if inducing_type == "random":
        idx = rng.choice(xy_all_s.shape[0], size=m, replace=(xy_all_s.shape[0] < m))
        return xy_all_s[idx]

    if inducing_type == "grid":
        if mask_geom_m is None:
            raise ValueError("grid requires mask_geom_m (NC polygon union) in meters CRS.")

        # build grid in meters bbox, then clip to NC, then standardize
        mins_m = xy_all_m.min(axis=0)
        maxs_m = xy_all_m.max(axis=0)
        span_m = np.maximum(maxs_m - mins_m, 1e-12)

        pad = grid_padding * span_m
        lo = mins_m - pad
        hi = maxs_m + pad

        side = int(np.ceil(np.sqrt(m * grid_oversample)))
        xs = np.linspace(lo[0], hi[0], side)
        ys = np.linspace(lo[1], hi[1], side)
        X, Y = np.meshgrid(xs, ys)
        grid_m = np.column_stack([X.ravel(), Y.ravel()])

        pg = prep(mask_geom_m)
        keep = np.array([pg.contains(Point(x, y)) for x, y in grid_m])
        grid_m = grid_m[keep]

        if grid_m.shape[0] < m:
            raise ValueError(
                f"Masked grid produced {grid_m.shape[0]} points (< m={m}). "
                f"Increase grid_oversample or grid_padding."
            )

        idx = rng.choice(grid_m.shape[0], size=m, replace=False)
        Z_m = grid_m[idx]
        return (Z_m - mu) / sd

    if inducing_type == "min_separation":
        if min_sep is None:
            raise ValueError("min_separation requires min_sep to be specified.")

        return select_inducing_with_min_sep(
            xy_all_s, m=m, min_sep=min_sep, seed=seed, max_passes=max_passes
        )

    raise ValueError("inducing_type must be one of: 'grid', 'kmeans', 'random', 'min_separation'")


class LGCP_Model(gpytorch.models.ApproximateGP):
    def __init__(
        self,
        inducing_points,
        name_prefix="cox_gp_model",
        beta=1.0,
        learn_inducing_locations=True,
    ):
        self.name_prefix = name_prefix

        # Define the variational distribution and strategy of the GP
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )

        # Define model
        super().__init__(variational_strategy=variational_strategy)

        # Define mean and kernel
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.MaternKernel(
            nu=2.5, active_dims=[0, 1]
        )
        self.beta = beta

    def forward(self, coords):
        mean = self.mean_module(coords)
        covar = self.covar_module(coords)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def guide(self, observed_points, counts, population):
        alpha1_loc = pyro.param(
            self.name_prefix + ".alpha1_loc",
            observed_points.new_tensor(-0.5),
        )
        alpha1_scale = pyro.param(
            self.name_prefix + ".alpha1_scale",
            observed_points.new_tensor(0.5),
            constraint=dist.constraints.positive,
        )
        pyro.sample(self.name_prefix + ".alpha1", dist.Normal(alpha1_loc, alpha1_scale))

        alpha0_loc = pyro.param(
            self.name_prefix + ".alpha0_loc",
            observed_points.new_tensor(-1.0),
        )
        alpha0_scale = pyro.param(
            self.name_prefix + ".alpha0_scale",
            observed_points.new_tensor(1.0),
            constraint=dist.constraints.positive,
        )
        pyro.sample(self.name_prefix + ".alpha0", dist.Normal(alpha0_loc, alpha0_scale))

        function_distribution = self.pyro_guide(observed_points, beta=self.beta)

        # Sample from q(x) at observed_points
        with pyro.plate(self.name_prefix + ".times_plate", dim=-1):
            pyro.sample(self.name_prefix + ".function_samples", function_distribution)

    def model(self, observed_points, counts, population):
        pyro.module(self.name_prefix + ".gp", self)
        function_distribution = self.pyro_model(observed_points, beta=self.beta)
        with pyro.plate(self.name_prefix + ".times_plate", dim=-1):
            function_samples = pyro.sample(
                self.name_prefix + ".function_samples", function_distribution
            )

        loc0  = observed_points.new_tensor(-1.0)
        scale0 = observed_points.new_tensor(1.0)
        loc1  = observed_points.new_tensor(-0.5)
        scale1 = observed_points.new_tensor(0.5)
        alpha0 = pyro.sample(self.name_prefix + ".alpha0", dist.Normal(loc0, scale0))
        alpha1 = pyro.sample(self.name_prefix + ".alpha1", dist.Normal(loc1, scale1))
        
        log_lambda = alpha0 + alpha1 * function_samples
        lambda_ = log_lambda.exp()
        rate = lambda_ * population
        with pyro.plate(self.name_prefix + ".counts", dim=-1):
            pyro.sample(self.name_prefix + ".observed", dist.Poisson(rate), obs=counts)
        self.arrival_intensity_samples = rate


def build_lgcp(
    areal_units_all: gpd.GeoDataFrame,
    df_obs: gpd.GeoDataFrame,
    m: int,
    seed: int,
    dtype: torch.dtype,
    *,
    inducing_type: InducingType = "kmeans",
    grid_padding: float = 0.02,
    grid_oversample: float = 2.5,
    kmeans_batch_size: int = 4096,
    min_sep: Optional[float] = None,
    max_passes: int = 3,
    beta: float = 0.01,
    learn_inducing: bool = False,
) -> Tuple[object, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """
    Builds the LGCP model and prepares the data for training and testing.
    Returns:
      lgcp_model, inducing_points, test_X_lgcp, train_X_lgcp, mu_xy, sd_xy

    inducing_points are returned in standardized space (same as test_X_lgcp/train_X_lgcp).
    """

    g_all = areal_units_all.to_crs("EPSG:32119")
    centroids_all = g_all.geometry.centroid
    xy_all_m = np.c_[centroids_all.x.to_numpy(), centroids_all.y.to_numpy()]
    mu = xy_all_m.mean(axis=0)
    sd = xy_all_m.std(axis=0)
    xy_all_s = (xy_all_m - mu) / sd
    test_X_lgcp = torch.tensor(xy_all_s, dtype=dtype)

    g_obs = gpd.GeoDataFrame(
        df_obs,
        geometry=gpd.points_from_xy(df_obs.lon, df_obs.lat),
        crs="EPSG:4326",
    ).to_crs("EPSG:32119")
    xy_obs_m = np.c_[g_obs.geometry.x.to_numpy(), g_obs.geometry.y.to_numpy()]
    train_X_lgcp = torch.tensor((xy_obs_m - mu) / sd, dtype=dtype)

    # inducing points
    mask_geom_m = None
    if inducing_type == "grid":
        mask_geom_m = geoseries_union_all(g_all.geometry)

    Z_s = initialize_inducing_points(xy_all_m=xy_all_m, m=m,inducing_type=inducing_type, seed=seed,
        mask_geom_m=mask_geom_m, grid_padding=grid_padding, grid_oversample=grid_oversample,
        kmeans_batch_size=kmeans_batch_size, min_sep=min_sep, max_passes=max_passes,
    )
    inducing_points = torch.tensor(Z_s, dtype=dtype)

    # make lgcp model
    pyro.clear_param_store()
    lgcp_model = LGCP_Model(inducing_points, beta=beta, learn_inducing_locations=learn_inducing)

    return lgcp_model, inducing_points, test_X_lgcp, train_X_lgcp, mu, sd


def attach_lgcp_outputs(areal_units_all: gpd.GeoDataFrame, lgcp_model, test_X_lgcp: torch.Tensor,
) -> gpd.GeoDataFrame:
    with torch.no_grad():
        f_post = lgcp_model(test_X_lgcp)
        latent_mean = f_post.mean.detach().cpu()
        alpha0 = pyro.param("cox_gp_model.alpha0_loc").detach().cpu()
        alpha1 = pyro.param("cox_gp_model.alpha1_loc").detach().cpu()
        log_lambda_mean = alpha0 + alpha1 * latent_mean

    out = areal_units_all.copy()
    out["latent_mean"] = latent_mean.detach().cpu().numpy()
    out["log_lambda_mean"] = log_lambda_mean.detach().cpu().numpy()
    return out


def make_kernel(which: str):
    """
    Create a kernel based on the specified type (input: str).
    """
    matern_ll = MaternKernel(
        nu=1.5,
        ard_num_dims=2, 
        active_dims=[0, 1],
    )

    matern_cov = MaternKernel(nu=1.5, ard_num_dims=2, active_dims=[2, 3])
    matern_cov_ll = MaternKernel(
        nu=1.5,
        ard_num_dims=4,
        active_dims=[2, 3, 0, 1],
    )
    k_latent = RBFKernel(active_dims=[4])

    cov_term = ScaleKernel(
        matern_cov,
        outputscale_prior=GammaPrior(2.0, 2.0),
        outputscale_constraint=Interval(0.2, 3.0),
    )
    cov_latent_term = ScaleKernel(
        ProductKernel(matern_cov, k_latent),
        outputscale_prior=GammaPrior(1.0, 2.0),
        outputscale_constraint=Interval(1e-4, 5.0)
    )
    spatial_term = ScaleKernel(
        matern_ll,
        outputscale_prior=GammaPrior(1.5, 3.0),
        outputscale_constraint=Interval(0.0, 2.0),
    )

    if which == "matern_cov":
        base = ScaleKernel(matern_cov_ll)
    elif which == "matern_ll":
        base = ScaleKernel(matern_ll)
    elif which == "cov_mult_latent":
        base = ScaleKernel(matern_cov_ll * k_latent)
    elif which == "cov_plus_latent":
        base = ScaleKernel(matern_cov_ll) + ScaleKernel(k_latent)
    elif which == "cov_plus_mult_latent":
        base = ScaleKernel(matern_cov_ll) + ScaleKernel(matern_cov_ll * k_latent)
    elif which == "cov_plus_mult_latent_plus_coords":
        base = cov_term + cov_latent_term + spatial_term
    elif which == "ll_plus_ll_mult_latent":
        base = ScaleKernel(matern_ll) + ScaleKernel(matern_ll * k_latent)

    else:
        raise ValueError(f"Unknown kernel spec: {which}")

    return base


def _print_base_kernel_info(prefix, k):
    """Print lengthscale + active_dims for a single base kernel."""
    cls_name = k.__class__.__name__
    print(f"{prefix}kernel: {cls_name}")
    if hasattr(k, "active_dims") and k.active_dims is not None:
        print(f"{prefix}  active_dims:", list(k.active_dims))
    if hasattr(k, "lengthscale"):
        ls = k.lengthscale.detach().cpu().numpy().ravel()
        print(f"{prefix}  lengthscale:", ls)


def summarize_kernel(model):
    """
    Summarize the learned hyperparameters of a GP model's covar_module, e.g. outputscale, lengthscale.
    """
    covar = model.covar_module
    print("\n=== Kernel summary ===")
    print("Full covar_module:\n", covar, "\n")

    if isinstance(covar, ScaleKernel):
        print("Top-level: ScaleKernel")
        print("  outputscale:", covar.outputscale.item())
        base = covar.base_kernel

        if isinstance(base, MaternKernel):
            _print_base_kernel_info("  ", base)

        elif isinstance(base, ProductKernel):
            print("  ProductKernel factors:")
            for i, k in enumerate(base.kernels):
                label = "covariance" if i == 0 else "latent"
                print(f"    [factor {i} – {label}]")
                _print_base_kernel_info("      ", k)
        else:
            print("  Unhandled base kernel type:", type(base))

    elif isinstance(covar, AdditiveKernel):
        print("Top-level: AdditiveKernel with", len(covar.kernels), "terms.")
        for i, term in enumerate(covar.kernels):
            print(f"\n[Additive term {i}]")
            if isinstance(term, ScaleKernel):
                print("  outputscale:", term.outputscale.item())
                base = term.base_kernel
            else:
                base = term

            if isinstance(base, MaternKernel):
                _print_base_kernel_info("  ", base)

            elif isinstance(base, RBFKernel):
                _print_base_kernel_info("  ", base)

            elif isinstance(base, ProductKernel):
                print("  ProductKernel factors:")
                for j, k in enumerate(base.kernels):
                    flabel = "covariance" if j == 0 else "latent"
                    print(f"    [factor {j} – {flabel}]")
                    _print_base_kernel_info("      ", k)
            else:
                print("  Unhandled base kernel type:", type(base))

    elif isinstance(covar, ProductKernel):
        print("Top-level: ProductKernel with", len(covar.kernels), "factors.")
        for i, k in enumerate(covar.kernels):
            print(f"\n[Product factor {i}]")
            base = k.base_kernel if isinstance(k, ScaleKernel) else k
            if isinstance(k, ScaleKernel):
                print("  outputscale:", k.outputscale.item())
            _print_base_kernel_info("  ", base)

    else:
        print("Unhandled covar_module type:", type(covar))
