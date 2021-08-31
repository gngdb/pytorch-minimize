from pytorch_minimize.optim import DualAnnealingWrapper

import math
import torch
import torch.nn as nn

def test_da(double=True, disp=False):
    class Ackley(nn.Module):
        def __init__(self):
            super().__init__()
            self.x = nn.Parameter(torch.ones(2))

        def forward(self):
            x = self.x
            arg1 = -0.2 * torch.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2)+1e-3)
            arg2 = 0.5 * (torch.cos(2. * math.pi * x[0]) + torch.cos(2. * math.pi * x[1]))
            return -20. * torch.exp(arg1) - torch.exp(arg2) + 20. + math.e

    bounds = [(-5, 5), (-5, 5)]
    da_kwargs = dict(bounds=bounds)
    minimizer_args = dict(method='SLSQP', options={'disp':disp, 'maxiter':10000})
    ackley = Ackley()
    if double:
        ackley = ackley.double()
    optimizer = DualAnnealingWrapper(ackley.parameters(), minimizer_args, da_kwargs)

    def closure():
        optimizer.zero_grad()
        loss = ackley()
        loss.backward()
        return loss

    optimizer.step(closure)

    print(optimizer.res.x, optimizer.res.fun)

if __name__ == '__main__':
    test_da(disp=True)
