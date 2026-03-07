"""
This script contains the code to implement the SVGPR+PS model, using a combination
of GPyTorch and Pyro.
Author: Zach Calhoun, Ellie Kim
Date: 02/27/2025
"""

import torch
import gpytorch
import geopandas as gpd
from shapely.geometry import Point
import pyro
import numpy as np
import pyro.distributions as dist
from sklearn.cluster import KMeans


def _clamp_sigma(sigma, floor=1e-3, ceil=None):
    if ceil is None:
        return sigma.clamp_min(floor)
    return sigma.clamp(min=floor, max=ceil)

class SVGPR_PS_BASE_NEW(gpytorch.models.ApproximateGP):
    """
    Define the base class for the preferential sampling model.
    """

    def __init__(
        self,
        num_points,
        area=None,
        inducing_points=None,
        inducing_point_prior=None,
        name_prefix="cox_gp_model",
        learn_inducing_locations=False,
        mean_temp=None,
        beta=1.0,
    ):
        self.name_prefix = name_prefix
        self.area = area

        if self.area is not None:
            self.mean_intensity = num_points / (self.area[0] * self.area[1])
        else:
            self.mean_intensity = None
        self.beta = beta

        self.mean_temp = mean_temp
        self.ps_scale = 1.0

        if inducing_points is None:
            raise ValueError("You must pass explicit inducing_points shaped like [num_inducing, 4].")

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=len(inducing_points)
        )

        if inducing_point_prior is not None:
            variational_distribution.initialize_variational_distribution(
                inducing_point_prior
            )

        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )

        super().__init__(variational_strategy=variational_strategy)

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=0.5,
                active_dims=[0, 1],
            )
        )

        self.covar_module.base_kernel.initialize(lengthscale=1.0)
        self.covar_module.outputscale = 1.551062822341919
        self.likelihood_noise = 1e-4 

    def forward(self, x):
        """The standard forward pass for GPs"""
        mean = self.mean_module(x)
        covar = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def model(self, points, inducing_points, y):
        """
        Define the model here
        """
        raise NotImplementedError("This method should be implemented in a subclass.")

class SVGPR_PS_MLE_NEW(SVGPR_PS_BASE_NEW):
    """
    This class defines the SVGPR-PS model, in which the model parameters are
    simply parameters, without a prior. This model learns the MLE of the model
    parameters (i.e. we are not learning a distribution over the parameters).
    """

    def model(self, points, inducing_points, quadrature_points, y, sigma, pop_dens_points, pop_dens_quad, counts_points): # sigma input = per-unit std from kriged variance
        """
        A simple model with parameters to learn (and no priors).
        """

        w = pyro.param(self.name_prefix + ".linear_weights", 
            torch.tensor([-0.5823, 1.0096])
        )
        b0 = torch.tensor([float(self.mean_temp) if self.mean_temp is not None else 0.0])
        b = pyro.param(self.name_prefix + ".linear_bias", b0)

        X_covariates = points[:, [2, 3]]
        linear_mean = X_covariates @ w + b

        pyro.module(self.name_prefix + ".gp", self)

        function_distribution = self.pyro_model(
            torch.vstack([points, quadrature_points]),
        )

        with pyro.plate(self.name_prefix + ".times_plate", dim=-1):
            function_samples = pyro.sample(
                self.name_prefix + ".function_samples", function_distribution
            )

        f_points, f_quad = function_samples.split(
            [points.size(0), quadrature_points.size(0)], dim=-1
        )

        predicted_y = f_points + linear_mean
        sigma = _clamp_sigma(sigma)

        with pyro.plate(self.name_prefix + ".observed_data", dim=-1):
            pyro.sample(
                self.name_prefix + ".observed",
                dist.Normal(predicted_y, sigma),
                obs=y,
            )

        alpha0 = pyro.sample(self.name_prefix + ".alpha0", dist.Normal(-1.0, 1.0))
        alpha1 = pyro.sample(self.name_prefix + ".alpha1", dist.Normal(-0.5, 0.5))
        eps = 1e-12

        pop_dens_points = pop_dens_points.to(device=points.device, dtype=points.dtype)
        pop_dens_quad   = pop_dens_quad.to(device=points.device, dtype=points.dtype)
        counts_points = counts_points.to(device=points.device, dtype=points.dtype)

        log_p_points = torch.log(pop_dens_points + eps)
        log_p_quad   = torch.log(pop_dens_quad   + eps)

        log_lambda_points = alpha0 + alpha1 * f_points + log_p_points
        log_lambda_quad   = alpha0 + alpha1 * f_quad   + log_p_quad

        quad_intensity_samples    = torch.exp(log_lambda_quad)

        arrival_log_intensities = (counts_points * log_lambda_points).sum(dim=-1)
        est_num_arrivals = self.est_num_arrivals(quad_intensity_samples)
        log_likelihood = arrival_log_intensities - est_num_arrivals
        
        pyro.factor(self.name_prefix + ".log_likelihood", self.ps_scale * log_likelihood)

    def est_num_arrivals(self, quad_intensity_samples):
        """
        Compute the expected number of arrivals. This method only works for a rectangular area case.
        Since we're working with polygons, this method will be overriden.
        """
        if self.area is None:
            raise ValueError("area is None; override est_num_arrivals in subclass.")
        return quad_intensity_samples.mean(dim=-1) * self.area[0] * self.area[1]
    
    def guide(self, points, inducing_points, quadrature_points, y, sigma, pop_dens_points, pop_dens_quad, counts_points):
        alpha1_loc = pyro.param(self.name_prefix + ".alpha1_loc", torch.tensor(-0.5))
        alpha1_scale = pyro.param(
            self.name_prefix + ".alpha1_scale",
            torch.tensor(1.0),
            constraint=dist.constraints.positive,
        )
        pyro.sample(self.name_prefix + ".alpha1", dist.Normal(alpha1_loc, alpha1_scale))

        alpha0_loc = pyro.param(self.name_prefix + ".alpha0_loc", torch.tensor(-1.0))
        alpha0_scale = pyro.param(
            self.name_prefix + ".alpha0_scale",
            torch.tensor(1.0),
            constraint=dist.constraints.positive,
        )
        pyro.sample(self.name_prefix + ".alpha0", dist.Normal(alpha0_loc, alpha0_scale))

        function_distribution = self.pyro_guide(torch.vstack([points, quadrature_points]))

        with pyro.plate(self.name_prefix + ".times_plate", dim=-1):
            pyro.sample(self.name_prefix + ".function_samples", function_distribution)


class SVGPR_PS_BLOCKS_NEW(SVGPR_PS_MLE_NEW):
    def __init__(
        self,
        num_points,
        df,
        inducing_point_prior=None,
        inducing_points=None,
        name_prefix="cox_gp_model",
        learn_inducing_locations=True,
        mean_temp=None,
    ):
        self.name_prefix = name_prefix
        self.df = df
        self.mean_intensity = num_points / self.df["population"].sum()

        self.area_quad = torch.tensor(df["ALAND"].to_numpy(), dtype=torch.float32)
        self.pop_dens_quad = torch.tensor((df["population"] / df["ALAND"]).to_numpy(), dtype=torch.float32)

        if inducing_points is None:
            raise AssertionError(
                "You must pass explicit inducing_points shaped like [num_inducing, 4]."
            )

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=len(inducing_points)
        )
        if inducing_point_prior is not None:
            variational_distribution.initialize_variational_distribution(
                inducing_point_prior
            )

        super().__init__(
            num_points,
            area=None,
            inducing_points=inducing_points,
            inducing_point_prior=inducing_point_prior,
            name_prefix=name_prefix,
            learn_inducing_locations=learn_inducing_locations,
            mean_temp=mean_temp,
        )

        self.total_area = torch.tensor(df["ALAND"].sum(), dtype=torch.float32)

    def est_num_arrivals(self, quad_intensity_samples):
        area_quad = self.area_quad.to(device=quad_intensity_samples.device,
                                dtype=quad_intensity_samples.dtype)
        return (quad_intensity_samples * area_quad).sum(dim=-1)

