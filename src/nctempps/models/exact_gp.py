from typing import Optional, Dict, Any, List
import gpytorch
import torch
import pandas as pd


# make sure json is safe (fails if user forgets a parameter)
def _require(cfg: dict, *keys: str):
    missing = [k for k in keys if k not in cfg]
    if missing:
        raise ValueError(f"ls_config type='{cfg.get('type')}' missing keys: {missing}. Got: {cfg}")


def _set_lengthscale_prior_or_constraint(base_kernel, *, ls_config: Optional[Dict[str, Any]]):
    """
    Modifies latent kernel in place to apply a lengthscale prior and/or constraint.

    ls_config examples:
      None  -> do nothing
      {"type": "none"}
      {"type": "interval", "min": 0.1, "max": 10.0}
      {"type": "smoothed_box", "min": 0.1, "max": 10.0, "sigma": 0.1}
      {"type": "half_normal", "scale": 5.0}
      {"type": "log_normal", "loc": 0.0, "scale": 1.0}   # log-lengthscale ~ N(loc, scale)
    """
    if not ls_config or ls_config.get("type", "none") == "none":
        return base_kernel

    t = ls_config["type"].lower()

    # hard constraints
    if t == "interval":
        _require(ls_config, "min", "max")
        base_kernel.register_constraint(
                "raw_lengthscale",
                gpytorch.constraints.Interval(ls_config["min"], ls_config["max"])
            )
        return base_kernel

    # priors
    if t == "smoothed_box":
        _require(ls_config, "min", "max")
        prior = gpytorch.priors.SmoothedBoxPrior(
            a=ls_config["min"], b=ls_config["max"], sigma=ls_config.get("sigma", 0.1)
        )
        base_kernel.register_prior(
            "lengthscale_prior",
            prior,
            lambda m: m.lengthscale,
            lambda m, v: m._set_lengthscale(v),
        )
        return base_kernel

    if t == "half_normal":
        _require(ls_config, "scale")
        # gpytorch uses torch distributions
        prior = gpytorch.priors.torch_priors.HalfNormalPrior(scale=ls_config["scale"])
        base_kernel.register_prior(
            "lengthscale_prior",
            prior,
            lambda m: m.lengthscale,
            lambda m, v: m._set_lengthscale(v),
        )
        return base_kernel

    if t == "log_normal":
        _require(ls_config, "loc", "scale")
        prior = gpytorch.priors.LogNormalPrior(
            loc=ls_config["loc"], scale=ls_config["scale"]
        )
        base_kernel.register_prior(
            "lengthscale_prior",
            prior,
            lambda m: m.lengthscale,
            lambda m, v: m._set_lengthscale(v),
        )
        return base_kernel

    raise ValueError(f"Unknown lengthscale prior/constraint type: {t}")



def make_latent_kernel(*, kernel_name: str, active_dim: int, ls_config: Optional[Dict[str, Any]] = None):
    """
    Creates a 1D latent kernel (active_dims=[active_dim]) based on kernel_name.
    Applies optional lengthscale prior/constraint via ls_config.
    """
    name = kernel_name.lower()

    if name == "rbf":
        k = gpytorch.kernels.RBFKernel(active_dims=[active_dim])
        return _set_lengthscale_prior_or_constraint(k, ls_config=ls_config)

    if name in ("matern0.5", "matern_0.5", "matern0p5"):
        k = gpytorch.kernels.MaternKernel(nu=0.5, active_dims=[active_dim])
        return _set_lengthscale_prior_or_constraint(k, ls_config=ls_config)

    if name in ("matern1.5", "matern_1.5", "matern1p5"):
        k = gpytorch.kernels.MaternKernel(nu=1.5, active_dims=[active_dim])
        return _set_lengthscale_prior_or_constraint(k, ls_config=ls_config)

    if name in ("matern2.5", "matern_2.5", "matern2p5"):
        k = gpytorch.kernels.MaternKernel(nu=2.5, active_dims=[active_dim])
        return _set_lengthscale_prior_or_constraint(k, ls_config=ls_config)

    if name == "linear":
        k = gpytorch.kernels.LinearKernel(active_dims=[active_dim])
        return k

    raise ValueError(f"Unknown latent kernel_name: {kernel_name}")


@torch.no_grad()
def component_posterior_means_exactgp(model, likelihood, train_X, train_y, test_X):
    """
    Returns:
      total_mean: (N*,)
      comp_means: list of (N*,) one per additive kernel component
    """
    model.eval()
    likelihood.eval()

    idx = train_X.new_tensor(model.mean_cols, dtype=torch.long)
    m_train = model.mean_module(train_X.index_select(dim=-1, index=idx))
    y_centered = (train_y - m_train).unsqueeze(-1)

    K = model.covar_module(train_X)
    sigma2 = likelihood.noise
    K_noise = K.add_diagonal(sigma2)

    alpha = K_noise.solve(y_centered)

    idx2 = test_X.new_tensor(model.mean_cols, dtype=torch.long)
    m_test = model.mean_module(test_X.index_select(dim=-1, index=idx2))
    K_star = model.covar_module(test_X, train_X)
    total = m_test + (K_star @ alpha).squeeze(-1)

    if not hasattr(model.covar_module, "kernels"):
        raise ValueError("Model covar_module doesn't look additive (no .kernels list).")

    comp_means = []
    for ki in model.covar_module.kernels:
        Ki_star = ki(test_X, train_X)
        comp_i = (Ki_star @ alpha).squeeze(-1)
        comp_means.append(comp_i)

    return total, comp_means



class ExactGPModel(gpytorch.models.ExactGP):
    """
    ExactGP: mean is LinearMean over [elev, t_s], potentially also containing the latent if
       latent_in_mean=True.
    Kernel is spatial, optionally combined with a latent kernel additively or multiplicatively.
    Only one of the following can be true: latent_in_cov, latent_in_mean.
    1) latent_in_mean=True: we naively apply preferential sampling correction
    2) latent_in_cov=True: we assume temperature should only go up if two locations are close spatially AND have similar latent values.
    
    """
    def __init__(self, train_x, train_y, likelihood, *, mean_cols: Optional[List[int]] = None,
                latent_in_cov: bool, latent_col: int = 4,
                latent_kernel: str = "rbf", latent_ls_config: Optional[Dict[str, Any]] = None,
                latent_combine: str = "sum_product",
    ):
        super().__init__(train_x, train_y, likelihood)

        self.mean_cols = list(mean_cols)
        self.mean_module = gpytorch.means.LinearMean(input_size=len(self.mean_cols))
        self.mean_module.weights.requires_grad = True

        matern_ll = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=2, active_dims=[0, 1])
        k_spatial = gpytorch.kernels.ScaleKernel(matern_ll)
        covar = k_spatial

        # optional latent kernel
        self.latent_in_cov = latent_in_cov
        self.latent_col = latent_col
        if latent_ls_config is not None and not latent_in_cov:
            print("[WARN] latent_ls_config was provided but latent_in_cov=False. Config will be ignored.")
        elif latent_combine is not None and not latent_in_cov:
            print("[WARN] latent_combine was provided but latent_in_cov=False. Kernel config (e.g. sum_product) will be ignored.")
        self.latent_term_id: Optional[int] = None

        if latent_in_cov:
            k_latent_base = make_latent_kernel(
                kernel_name=latent_kernel,
                active_dim=latent_col,
                ls_config=latent_ls_config,
            )
            combine = latent_combine.lower().strip()

            if combine == "add":
                # K = K_spatial + K_latent
                covar = gpytorch.kernels.ScaleKernel(matern_ll) + gpytorch.kernels.ScaleKernel(k_latent_base)
                self.latent_term_id = 1
            elif combine in ("product", "mult"):
                # K = K_spatial * K_latent
                covar = gpytorch.kernels.ScaleKernel(matern_ll * k_latent_base)
                self.latent_term_id = 0
            elif combine in ("sum_product", "add_product"):
                # K = K_spatial + (K_spatial * K_latent)
                matern1 = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=2, active_dims=[0, 1])
                matern2 = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=2, active_dims=[0, 1])
                covar = gpytorch.kernels.ScaleKernel(matern1) + gpytorch.kernels.ScaleKernel(matern2 * k_latent_base)
                self.latent_term_id = 1
            elif combine in ("cov_plus_mult_latent_plus_coords"):
                # K = K_spatial + (K_spatial * K_latent) + spatial_term
                matern_cov = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=2, active_dims=[2, 3]) # elev and t_s
                covar = gpytorch.kernels.ScaleKernel(matern_cov) + gpytorch.kernels.ScaleKernel(matern_cov * k_latent_base) + gpytorch.kernels.ScaleKernel(matern_ll)
                self.latent_term_id = 1
            elif combine in ("latent_local_and_global"):
                # K = K_spatial + (K_spatial * K_latent) + K_latent
                matern1 = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=2, active_dims=[0, 1])
                matern2 = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=2, active_dims=[0, 1])
                k_latent1 = make_latent_kernel(kernel_name=latent_kernel, active_dim=latent_col, ls_config=latent_ls_config)
                k_latent2 = make_latent_kernel(kernel_name=latent_kernel, active_dim=latent_col, ls_config=latent_ls_config)
                covar = gpytorch.kernels.ScaleKernel(matern1) + gpytorch.kernels.ScaleKernel(matern2 * k_latent1) + gpytorch.kernels.ScaleKernel(k_latent2)
            else:
                raise ValueError(
                    f"Unknown latent_combine='{latent_combine}'. "
                    "Use one of: 'add', 'product', 'sum_product', 'latent_local_and_global', 'cov_plus_mult_latent_plus_coords'."
                )

        self.covar_module = covar

    def forward(self, x):
        mean_inputs = x[:, self.mean_cols]
        mean_x = self.mean_module(mean_inputs)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def latent_component_column(self, tracts_pred: pd.DataFrame) -> str:
        if not self.latent_in_cov:
            raise ValueError("latent_in_cov=False; no latent component column exists.")
        if self.latent_term_id is None:
            raise ValueError("latent_term_id is None; did you build the model with a latent term?")

        prefix = f"exact_gp_comp{self.latent_term_id}_"
        matches = [c for c in tracts_pred.columns if c.startswith(prefix)]
        if not matches:
            raise ValueError(
                f"Expected a column starting with {prefix!r}, but none found. "
                "Did you attach the decomposition columns to tracts_pred?"
            )
        return matches[0]