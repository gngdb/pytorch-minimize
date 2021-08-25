import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_minimize.optim import MinimizeWrapper

def test_index_bug():
    torch.manual_seed(0)
    params = {'a': torch.randn(10), 'b': torch.randn(9), 'c': torch.randn(8)}
    params = list(params.values())
    minimizer_args = dict(method='CG', options={'disp':True, 'maxiter':100})
    optimizer = MinimizeWrapper(params, minimizer_args)

    _params = optimizer.np_unravel_unpack(optimizer.ravel_pack(params))
    for p, _p in zip(params, _params):
        assert torch.abs(p-_p).max() < 1e-5

if __name__ == '__main__':
    test_index_bug()
