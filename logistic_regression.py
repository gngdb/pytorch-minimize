from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from optim import MinimizeWrapper


class LogReg(nn.Module):
    def __init__(self):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(28*28, 10)

    def forward(self, x):
        n = x.size(0)
        x = self.fc(x.view(n,-1))
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, dataset, optimizer, epoch):
    model.train()
    data, target = dataset
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
    print(f"Step: {epoch} Loss: {closure.loss:.2f}")

def test(model, device, dataset):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        data, target =  dataset
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='mean').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data),
        100. * correct / len(data)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # train_kwargs = {'batch_size': 50000} # all of MNIST
    # test_kwargs = {'batch_size': 10000} # all of MNIST
    train_kwargs = {'batch_size': 500} 
    test_kwargs = {'batch_size': 100} 
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    train_dataset = next(iter(train_loader))
    test_dataset = next(iter(test_loader))

    model = LogReg().to(device)
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    minimizer_args = dict(method="CG", options={'disp':True})
    # minimizer_args = dict(method="Newton-CG", options={'disp':True})
    # minimizer_args = dict(method="L-BFGS-B", options={'disp':True})
    # minimizer_args = dict(method="BFGS", options={'disp':True})
    optimizer = MinimizeWrapper(model.parameters(), minimizer_args)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_dataset, optimizer, epoch)
        test(model, device, test_dataset)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
