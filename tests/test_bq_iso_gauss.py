import numpy as np
import torch

from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel
from bqtorch.models.gp_regression import GP
from bqtorch.kernels.quadrature_kernels import QuadratureRBFGaussPrior

import pytest

@pytest.fixture
def create_bayesian_quadrature_iso_gauss():

    x1 = torch.from_numpy(np.array([[-1, 1], [0, 0], [-2, 0.1]]))
    x2 = torch.from_numpy(np.array([[-1, 1], [0, 0], [-2, 0.1], [-3, 3]]))
    M1 = x1.size()[0]
    M2 = x2.size()[0]
    D = x1.size()[1]

    prior_mean = torch.from_numpy(np.arange(D))[None, :]
    prior_variance = 2.

    rbf = RBFKernel()
    rbf.lengthscale = 1.
    kernel = ScaleKernel(rbf)
    kernel.outputscale = 1.

    bqkernel = QuadratureRBFGaussPrior(kernel, prior_mean, prior_variance)

    return bqkernel, x1, x2, M1, M2, D

def test_bq_iso_gauss_q_K(create_bayesian_quadrature_iso_gauss):

    bqkernel, x1, x2, M1, M2, D = create_bayesian_quadrature_iso_gauss

    qK = bqkernel.qK(x2).detach().numpy()[0,:]

    intervals = np.array([[0.28128128187888524, 0.2831094284574598],
                          [0.28135046180349665, 0.28307273575812275],
                          [0.14890780669545667, 0.15015321562978945],
                          [0.037853812661332246, 0.038507854167645676]])

    for i in range(4):
        assert intervals[i, 0] < qK[i] < intervals[i, 1]

def test_bq_iso_gauss_qKq(create_bayesian_quadrature_iso_gauss):

    bqkernel, x1, x2, M1, M2, D = create_bayesian_quadrature_iso_gauss

    interval = [0.19975038300858916, 0.20025772185633567]

    qKq = bqkernel.qKq().detach().numpy().squeeze()

    assert interval[0] < qKq < interval[1]