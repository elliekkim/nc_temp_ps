"""
Helpful training functions for LGCP models.

"""
from typing import List, Tuple, Optional, Dict, Any, Literal
import tqdm
from tqdm import tqdm
import pyro
import torch
import gpytorch
import matplotlib.pyplot as plt
import os
from nctempps.models.exact_gp import ExactGPModel
from pyro.optim import ClippedAdam


def init_linear_mean_ols(mean_module, X_full, y, mean_cols):
    '''
    Initialize the linear mean module using OLS estimates.
    Uses torch.linalg.lstsq to compute the OLS solution.
    Parameters:
    mean_module : gpytorch.means.LinearMean
        The linear mean module to initialize.
    X_full : torch.Tensor
        The full input feature tensor of shape (N, D).
    y : torch.Tensor
        The target tensor of shape (N,).
    mean_cols : List[int]
        The list of column indices to use for the linear mean.
    Returns
    -------
    None
    '''
    if mean_cols is None or len(mean_cols) == 0:
        raise ValueError("mean_cols must be a non-empty list of ints.")
    
    mean_cols = [int(c) for c in mean_cols]
    X = X_full[:, mean_cols]
    y = y.reshape(-1, 1).to(device=X.device, dtype=X.dtype)

    ones = torch.ones((X.shape[0], 1), device=X.device, dtype=X.dtype)
    Xd = torch.cat([ones, X], dim=1)

    beta = torch.linalg.lstsq(Xd, y).solution # pylint: disable=not-callable
    b0 = beta[0].squeeze()
    w  = beta[1:].squeeze(-1)

    with torch.no_grad():
        mean_module.bias.copy_(b0.to(mean_module.bias).reshape_as(mean_module.bias))
        mean_module.weights.copy_(w.to(mean_module.weights).reshape_as(mean_module.weights))


def train_exact_gp(output_dir, train_X: torch.Tensor, train_y: torch.Tensor, lr: float, num_iter: int, mean_cols: List[int], ols_init_mean: bool = False,
                   latent_in_cov: bool = False, latent_col: int = 4,
                   *, latent_kernel: str = "rbf", latent_ls_config: Optional[Dict[str, Any]] = None, latent_combine: str = "sum_product",
                    # scheduler controls:
                    lr_schedule: Literal["none", "cosine", "cosine_wr"] = "none",
                    eta_min: Optional[float] = None,
                    cosine_T_max: Optional[int] = None,
                    wr_T0: Optional[int] = None,
                    wr_Tmult: int = 2,
                    # logging
                    log_every: int = 10,
                    device: Optional[torch.device] = None,
                    dtype: Optional[torch.dtype] = None,
) -> Tuple[ExactGPModel, gpytorch.likelihoods.GaussianLikelihood, List[float]]:
    """
    Boiler plate code for training ExactGP models.
    Parameters:
    train_X : torch.Tensor
        Input coordinates of shape (N, D)
    train_y : torch.Tensor
        Observed values of shape (N,)
    lr : float
        Learning rate for the optimizer.
    num_iter : int
        Number of training iterations.
    lr_schedule:
      - "none": no scheduler
      - "cosine": CosineAnnealingLR
      - "cosine_wr": CosineAnnealingWarmRestarts
    Returns
    -------
    model : ExactGPModel
        Trained ExactGP model.
    likelihood : gpytorch.likelihoods.GaussianLikelihood
        Likelihood associated with the GP model.
    losses : list of float
        List of loss values at each iteration.
    """
    if device is None:
        device = train_X.device
    if dtype is None:
        dtype = train_X.dtype
        
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(
        train_X,
        train_y,
        likelihood,
        mean_cols=mean_cols,
        latent_in_cov=latent_in_cov,
        latent_col=latent_col,
        latent_kernel=latent_kernel,
        latent_ls_config=latent_ls_config,
        latent_combine=latent_combine
    )

    if ols_init_mean:
        init_linear_mean_ols(model.mean_module, train_X, train_y, mean_cols)

    train_X = train_X.to(device=device, dtype=dtype)
    train_y = train_y.to(device=device, dtype=dtype)
    model = model.to(device=device, dtype=dtype)
    likelihood = likelihood.to(device=device, dtype=dtype)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # create LR scheduler
    if eta_min is None:
        eta_min = lr * 0.05

    scheduler = None
    if lr_schedule == "none":
        scheduler = None

    elif lr_schedule == "cosine":
        if cosine_T_max is None:
            cosine_T_max = num_iter
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_T_max,
            eta_min=eta_min,
        )

    elif lr_schedule == "cosine_wr":
        if wr_T0 is None:
            wr_T0 = max(10, num_iter // 10)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=wr_T0,
            T_mult=wr_Tmult,
            eta_min=eta_min,
        )

    else:
        raise ValueError(f"Unknown lr_schedule={lr_schedule!r}")

    # training loop
    losses = []
    for i in range(num_iter):
        optimizer.zero_grad()
        output = model(train_X)
        loss = -mll(output, train_y)
        losses.append(float(loss.item()))
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            if lr_schedule == "cosine":
                scheduler.step()
            else:
                scheduler.step(i + 1)

        if log_every > 0 and (i % log_every) == 0:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"[ExactGP] iter {i:4d}/{num_iter}  loss={loss.item():.4f}  lr={cur_lr:.6g}")

    model.eval()
    likelihood.eval()
    return model, likelihood, losses


def train_pyro_model(model, X: torch.Tensor, counts: torch.Tensor, population: torch.Tensor,
               lr=0.01, num_iter=300, num_particles=32, lrd=0.99) -> List[float]:
    """
    Boiler plate code for training pyro-based models.

    Parameters:
    model : pyro-based model with .model and .guide methods
    X : torch.Tensor
        Input coordinates of shape (N, D)
    counts : torch.Tensor
        Observed counts of shape (N,)
    population : torch.Tensor
        Population of shape (N,)
    lr : float
        Learning rate for the optimizer.
    num_particles : int
        Number of particles for the ELBO estimator.
    num_iter : int
        Number of training iterations.
    lrd : float
        Learning rate decay factor.
    Returns
    -------
    losses : list of float
        List of loss values at each iteration.
    """
    losses = []
    pyro.clear_param_store()

    optimizer = ClippedAdam({"lr": lr})
    loss = pyro.infer.Trace_ELBO(
        num_particles=num_particles, vectorize_particles=True, retain_graph=True,
    )
    infer = pyro.infer.SVI(model.model, model.guide, optimizer, loss=loss)

    model.train()
    loader = tqdm(range(num_iter))
    for _ in loader:
        loss_val = infer.step(X, counts, population)
        losses.append(loss_val)
        loader.set_postfix(loss=loss_val)
    return losses