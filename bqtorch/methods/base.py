import torch
from torch import Tensor
from gpytorch.models.exact_prediction_strategies import DefaultPredictionStrategy

from bqtorch.kernels.quadrature_kernels import QuadratureRBFGaussPrior
from bqtorch.models.gp_regression import GP
from bqtorch.utils.optimize_helper import optimze_gp
from typing import Tuple


class BayesianQuadrature(object):

    def __init__(self, gp: GP) -> None:
        self.gp = gp

    def optimize(self, *args) -> None:
        """Optimize GP hyperparameters"""
        optimze_gp(self.gp, *args)

    def integrate(self) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError


class VanillaBQGaussPrior(BayesianQuadrature):

    def __init__(self, gp: GP, prior_mean, prior_variance) -> None:
        super().__init__(gp)
        self.wrapped_kernel = QuadratureRBFGaussPrior(self.gp.covar, prior_mean, prior_variance)

    def integrate(self) -> Tuple[Tensor, Tensor]:
        """Return integral mean and integral variance"""
        kernel_mean = self.wrapped_kernel.qK(self.gp.train_inputs[0])

        def get_mean_cache():
            train_prior_dist = self.gp(self.gp.train_inputs[0])
            prediction_strategy = DefaultPredictionStrategy(train_inputs=self.gp.train_inputs[0],
                                                            train_prior_dist=train_prior_dist,
                                                            train_labels=self.gp.train_targets,
                                                            likelihood=self.gp.likelihood)
            return prediction_strategy.mean_cache

        def get_zKz(z):
            likelihood = self.gp.likelihood
            train_prior_dist = self.gp(self.gp.train_inputs[0])
            train_x = self.gp.train_inputs[0]
            mvn = likelihood(train_prior_dist, train_x)
            train_train_covar = mvn.lazy_covariance_matrix
            Kz = train_train_covar.inv_matmul(z.t())
            return torch.matmul(z, Kz)

        mean_cache = get_mean_cache()
        integral_mean = torch.matmul(kernel_mean, mean_cache)
        integral_var = self.wrapped_kernel.qKq() - get_zKz(kernel_mean)
        return (integral_mean.squeeze(), integral_var.squeeze())


if __name__ == '__main__':
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.kernels import ScaleKernel, RBFKernel
    from bqtorch.models.gp_regression import GP

    rbf = ScaleKernel(RBFKernel())
    likelihood = GaussianLikelihood()
    likelihood.noise = 0.01
    prior_mean = torch.rand(1, 5)
    prior_variance = 1.
    x = torch.randn(5, 1)
    y = torch.randn(5, )
    gp = GP(x, y, rbf, likelihood)

    bq = VanillaBQGaussPrior(gp, prior_mean, prior_variance)
    bq.optimize()
    integral = bq.integrate()
    print(integral)
