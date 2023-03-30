import torch

import torch.nn as nn
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from qpth.qp import QPFunction

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('bmh')


class OptNet(nn.Module):
    def __init__(self, nFeatures, nHidden, nCls, bn, nineq=200, neq=0, eps=1e-4):
        super().__init__()
        self.nFeatures = nFeatures
        self.nHidden = nHidden
        self.bn = bn
        self.nCls = nCls
        self.nineq = nineq
        self.neq = neq
        self.eps = eps

        # Normal BN/FC layers.
        if bn:
            self.bn1 = nn.BatchNorm1d(nHidden)
            self.bn2 = nn.BatchNorm1d(nCls)

        self.fc1 = nn.Linear(nFeatures, nHidden)
        self.fc2 = nn.Linear(nHidden, nCls)

        # QP params.
        self.M = Variable(torch.tril(torch.ones(nCls, nCls)))
        self.L = Parameter(torch.tril(torch.rand(nCls, nCls)))
        self.G = Parameter(torch.Tensor(nineq,nCls).uniform_(-1,1))
        self.z0 = Parameter(torch.zeros(nCls))
        self.s0 = Parameter(torch.ones(nineq))

    def forward(self, x):
        nBatch = x.size(0)

        # Normal FC network.
        x = x.view(nBatch, -1)
        x = F.relu(self.fc1(x))
        if self.bn:
            x = self.bn1(x)
        x = F.relu(self.fc2(x))
        if self.bn:
            x = self.bn2(x)

        # Set up the qp parameters Q=LL^T and h = Gz_0+s_0.
        L = self.M*self.L
        Q = L.mm(L.t()) + self.eps*Variable(torch.eye(self.nCls))
        h = self.G.mv(self.z0)+self.s0
        e = Variable(torch.Tensor())
        x = QPFunction(verbose=-1)(Q, x, self.G, h, e, e)

        return F.log_softmax(x)


if __name__ == "__main__":
    # Create random data
    nBatch, nFeatures, nHidden, nCls = 16, 20, 20, 2
    x = Variable(torch.randn(nBatch, nFeatures), requires_grad=False)
    y = Variable((torch.rand(nBatch) < 0.5).long(), requires_grad=False)

    # Initialize the model.
    model = OptNet(nFeatures, nHidden, nCls, bn=False)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Initialize the optimizer.
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    for t in range(500):
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(x)

        # Compute and print loss.
        loss = loss_fn(y_pred, y)
        if t % 25 == 0:
            print('Iteration {}, loss = {:.2f}'.format(t, loss.item()))
        losses.append(loss.item())

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable weights
        # of the model)
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()

    plt.plot(losses)
    plt.ylabel('Cross Entropy Loss')
    plt.xlabel('Iteration')
    plt.ylim(ymin=0.)
