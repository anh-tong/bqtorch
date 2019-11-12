import torch
import torch.optim as optim
from gpytorch.mlls import ExactMarginalLogLikelihood
from bqtorch.models.gp_regression import GP

def optimze_gp(gp: GP, lr=0.01, epoch=300, verbose=True) -> None:

    optimizer = optim.Adam(gp.parameters(), lr=lr)
    train_x = gp.train_inputs[0]
    train_y = gp.train_targets
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    for i in range(epoch):
        gp.train()
        output = gp(train_x)
        loss = - mll(output, train_y)
        loss.backward()
        optimizer.step()
        if verbose:
            print("Iter: {} \t Negative Log Likelihood: {:.2f}".format(i+1, loss.item()))
