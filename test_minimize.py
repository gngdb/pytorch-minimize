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

def main(method):
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
        def __call__(self):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target) 
            loss.backward()
            self.loss = loss.item()
            return loss
    closure = Closure()
    optimizer.step(closure)

    # check if train loss is zero (overfitting)
    assert abs(closure.loss) < 1e-1, f"Train loss not near zero: {closure.loss}"
    return optimizer.res, closure.loss

def test_cg():
    return main("CG")

if __name__ == "__main__":
    res, loss = test_cg()
    # print(res)
    print(f"Train Loss: {loss:.2f}")

