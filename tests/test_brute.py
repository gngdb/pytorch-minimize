from pytorch_minimize.optim import BruteWrapper 

import math
import torch
import torch.nn as nn

def test_brute(double=True, disp=False):
    class Ackley(nn.Module):
        def __init__(self):
            super().__init__()
            self.x = nn.Parameter(torch.zeros(2))

        def forward(self):
            x = self.x
            arg1 = -0.2 * torch.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
            arg2 = 0.5 * (torch.cos(2. * math.pi * x[0]) + torch.cos(2. * math.pi * x[1]))
            return -20. * torch.exp(arg1) - torch.exp(arg2) + 20. + math.e

    #bounds = [(-5, 5), (-5, 5)]
    ranges = (slice(-5, 5, 0.25), slice(-5, 5, 0.25))
    brute_kwargs = dict(ranges=ranges, disp=disp)
    ackley = Ackley()
    if double:
        ackley = ackley.double()
    optimizer = BruteWrapper(ackley.parameters(), brute_kwargs)

    def closure():
        with torch.no_grad():
            return ackley()

    optimizer.step(closure)

    print(optimizer.res.x, optimizer.res.fun)

if __name__ == '__main__':
    test_brute(disp=True)
