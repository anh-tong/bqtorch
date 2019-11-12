import numpy as np

import torch
from torch import Tensor
from gpytorch.kernels import RBFKernel, ScaleKernel

class QuadratureKernel(object):

    """Wrap GPytorch kernel with kernel integration"""

    def __init__(self, kernel):
        self.kernel = kernel

    def qK(self, x):
        """Compute integration kernel mean"""
        raise NotImplementedError

    def qKq(self):
        """Compute the double integration """
        raise NotImplementedError

class QuadratureRBFGaussPrior(QuadratureKernel):

    """Integral kernel with isotropic Gaussian prior"""

    def __init__(self, kernel:ScaleKernel,
                 prior_mean: Tensor,
                 prior_variance: Tensor) -> None:
        """

        :param kernel: scaled RBF kernel
        :param prior_mean: size 1 x D
        :param prior_variance: size 1 x D
        """

        assert isinstance(kernel, ScaleKernel) is True
        assert isinstance(kernel.base_kernel, RBFKernel) is True
        super().__init__(kernel)
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.input_dim = self.prior_mean.size()[0]

    def qK(self, x):

        kernel_variance = self.kernel.outputscale
        kernel_lengthscale = self.kernel.base_kernel.lengthscale

        det_factor = (self.prior_variance / kernel_lengthscale**2 + 1.).pow(self.input_dim / 2.)
        scale = (kernel_lengthscale.pow(2) + self.prior_variance).sqrt()
        scaled_diff = (self.prior_mean - x) / (scale * np.sqrt(2))
        kernel_mean = (kernel_variance / det_factor) * torch.exp(-torch.sum(scaled_diff.pow(2), dim=1))
        return kernel_mean

    def qKq(self):

        kernel_variance = self.kernel.outputscale
        kernel_lengthscale = self.kernel.base_kernel.lengthscale

        det_factor = (self.prior_variance / kernel_lengthscale ** 2 + 1.).pow(self.input_dim / 2.)

        return kernel_variance / det_factor

if __name__ == '__main__':

    rbf = ScaleKernel(RBFKernel())
    prior_mean = torch.rand(1,5)
    prior_variance = torch.ones_like(prior_mean)
    quadrature = QuadratureRBFGaussPrior(rbf, prior_mean, prior_variance)
    x = torch.randn(5,5)
    qK = quadrature.qK(x)
    qKq = quadrature.qKq()
    print(qK, qKq)