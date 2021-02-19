import torch
import numpy as np
from scipy.optimize import minimize

class MinimizeWrapper(torch.optim.Optimizer):
    def __init__(self, params, minimizer_args):
        assert type(minimizer_args) is dict
        if 'jac' not in minimizer_args:
            minimizer_args['jac'] = True
        assert minimizer_args['jac'] in [True, False], \
                "separate jac function not supported"
        self.minimizer_args = minimizer_args
        if 'options' not in self.minimizer_args:
            self.minimizer_args.update({'options':{}})
        if 'maxiter' not in self.minimizer_args['options']:
            self.minimizer_args['options'].update({'maxiter':2})
        super(MinimizeWrapper, self).__init__(params, self.minimizer_args)
        assert len(self.param_groups) == 1, "only supports one group"

    def ravel_pack(self, tensors):
        # pack tensors into a numpy array
        def numpyify(tensor):
            if tensor.device != torch.device('cpu'):
                tensor = tensor.cpu()
            return tensor.detach().numpy()
        x = np.concatenate([numpyify(tensor).ravel() for tensor in tensors], 0)
        return x

    def unravel_unpack(self, x):
        # unpack parameters from a numpy array
        _group = next(iter(self.param_groups))
        _params = _group['params'] # use params as shape reference
        x = torch.from_numpy(x.astype(np.float32))
        i = 0
        params = []
        for _p in _params:
            j = _p.numel()
            p = x[i:i+j].view(_p.size())
            p = p.to(_p.device)
            params.append(p)
            i = j
        return params

    @torch.no_grad()
    def step(self, closure):
        group = next(iter(self.param_groups))
        params = group['params']
        # this check passes
        # _params = self.unravel_unpack(self.ravel_pack(params))
        # for p, _p in zip(params, _params):
        #     assert torch.abs(p-_p).max() < 1e-5

        def torch_wrapper(x):
            # monkey patch x back into the model parameters
            _params = self.unravel_unpack(x)
            for p, _p in zip(params, _params):
                p.data = _p
            with torch.enable_grad():
                loss = closure()
            if self.minimizer_args['jac']:
                grads = self.ravel_pack([p.grad for p in params])
                return loss, grads
            else:
                return loss

        # run the minimizer
        x0 = self.ravel_pack(params)
        res = minimize(torch_wrapper, x0, **self.minimizer_args)

        # set the final parameters
        _params = self.unravel_unpack(res.x)
        for p, _p in zip(params, _params):
            p.data = _p
