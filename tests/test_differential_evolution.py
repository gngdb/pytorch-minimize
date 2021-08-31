from pytorch_minimize.optim import DifferentialEvolutionWrapper

import math
import torch
import torch.nn as nn

def test_differential_evolution(double=True, disp=False):
    def ackley(x):
        arg1 = -0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
        arg2 = 0.5 * (np.cos(2. * np.pi * x[0]) + np.cos(2. * np.pi * x[1]))
        return -20. * np.exp(arg1) - np.exp(arg2) + 20. + np.e

    class Ackley(nn.Module):
        def __init__(self):
            super().__init__()
            self.x = nn.Parameter(torch.zeros(2))

        def forward(self):
            x = self.x
            arg1 = -0.2 * torch.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
            arg2 = 0.5 * (torch.cos(2. * math.pi * x[0]) + torch.cos(2. * math.pi * x[1]))
            return -20. * torch.exp(arg1) - torch.exp(arg2) + 20. + math.e

    bounds = [(-5, 5), (-5, 5)]
    de_kwargs = dict(bounds=bounds, disp=disp)
    #result = differential_evolution(ackley, bounds, disp=disp)
    ackley = Ackley()
    if double:
        ackley = ackley.double()
    optimizer = DifferentialEvolutionWrapper(ackley.parameters(), de_kwargs)

    def closure():
        with torch.no_grad():
            return ackley()

    optimizer.step(closure)

    print(optimizer.res.x, optimizer.res.fun)

if __name__ == '__main__':
    test_differential_evolution(disp=True)
