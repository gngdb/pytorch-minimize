from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from optim import MinimizeWrapper
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


n_samples = 120
n_features = 20
n_classes = 10


class LogReg(nn.Module):
    def __init__(self):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(n_features, n_classes)

    def forward(self, x):
        n = x.size(0)
        x = self.fc(x.view(n,-1))
        output = F.log_softmax(x, dim=1)
        return output

def main(method, disp=True):
    # only run tests on CPU
    device = torch.device('cpu')

    # seed everything
    torch.manual_seed(0)
    np.random.seed(0)

    # generate classification dataset
    X, y = make_classification(n_samples=n_samples,
                               n_informative=10,
                               n_features=n_features,
                               n_classes=n_classes)
    # split into training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y,
            test_size=(2./12.), random_state=0)
    def torchify(X, y):
        return torch.from_numpy(X).float(), torch.from_numpy(y).long()
    train_dataset = torchify(X_train, y_train)
    test_dataset = torchify(X_test, y_test)

    # test sklearn
    # clf = LogisticRegression(penalty='none').fit(X_train, y_train)
    # print(clf.score(X_train, y_train))
    # print(log_loss(y_train, clf.predict_proba(X_train)))

    # instance model
    model = LogReg().to(device)

    # instance optimizer
    minimizer_args = dict(method=method, options={'disp':True, 'maxiter':10000})
    optimizer = MinimizeWrapper(model.parameters(), minimizer_args)

    # train
    model.train()
    data, target = train_dataset
    data, target = data.to(device), target.to(device)
    class Closure():
        def __init__(self, model):
            self.model = model
        
        @staticmethod
        def loss(model):
            output = model(data)
            return F.nll_loss(output, target) 

        def __call__(self):
            optimizer.zero_grad()
            loss = self.loss(self.model)
            loss.backward()
            self._loss = loss.item()
            return loss
    closure = Closure(model)
    optimizer.step(closure)

    # check if train loss is zero (overfitting)
    assert abs(closure._loss) < 1e-1, f"Train loss not near zero with {method}: {closure._loss}"
    return optimizer.res, closure._loss

def test_jac_methods():
    # test methods that require only the jacobian and not the hessian
    methods = ["CG", "BFGS", "L-BFGS-B", "SLSQP"]
    failing_methods = ["TNC"]
    for method in methods:
        _ = main(method, disp=False)

def test_hess_methods():
    methods = ["Newton-CG", "trust-ncg", "trust-krylov", "trust-exact", "trust-constr"]
    failing_methods = ["dogleg"]
    for method in methods:
        _ = main(method, disp=False)

if __name__ == "__main__":
    res, loss = main("trust-constr")
    # print(res)
    print(f"Train Loss: {loss:.2f}")

