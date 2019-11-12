
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel
from gpytorch.distributions import MultivariateNormal

class GP(ExactGP):

    def __init__(self,
                 train_x,
                 train_y,
                 kernel,
                 likelihood
                 ):
        super().__init__(train_x, train_y, likelihood)
        self.mean = ConstantMean()
        self.covar = kernel

    def forward(self, x):
        mean_x = self.mean(x)
        covar_x = self.covar(x)
        return MultivariateNormal(mean_x, covar_x)
