import torch
import torch.nn as nn
from torch import optim

from graphgallery.nn.metrics import Accuracy
from graphgallery.nn.layers.pytorch import activations, GraphEigenConv, Sequential

from .base_sat import BaseSAT


class GCN(BaseSAT):
    def __init__(self,
                 in_features,
                 out_features,
                 alpha=None,  # unused
                 K=None,  # unused
                 eps_U=0.3,
                 eps_V=1.2,
                 lamb_U=0.8,
                 lamb_V=0.8,
                 hids=[],
                 acts=[],
                 dropout=0.5,
                 weight_decay=5e-4,
                 lr=0.01,
                 bias=False):

        super().__init__()

        conv = []
        conv.append(nn.Dropout(dropout))

        for hid, act in zip(hids, acts):
            conv.append(GraphEigenConv(in_features,
                                       hid,
                                       bias=bias))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_features = hid

        conv.append(GraphEigenConv(in_features, out_features, bias=bias))
        conv = Sequential(*conv)

        self.conv = conv
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam(self.parameters(),
                                          weight_decay=weight_decay, lr=lr),
                     metrics=[Accuracy()])
        self.eps_U = eps_U
        self.eps_V = eps_V
        self.lamb_U = lamb_U
        self.lamb_V = lamb_V

    def forward(self, x, U, V=None):
        """
        x: node attribute matrix
        if `V=None`:
            U: (N, N) adjacency matrix
        else:
            U: (N, k) eigenvector matrix
            V: (k,) eigenvalue
        """
        x = self.conv(x, U, V)
        return x
