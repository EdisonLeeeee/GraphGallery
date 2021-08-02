import torch
import torch.nn as nn
from torch import optim

from graphgallery.nn.metrics.pytorch import Accuracy
from graphgallery.nn.layers.pytorch import activations, SpectralEigenConv
from .base_sat import BaseSAT


class SSGC(BaseSAT):
    def __init__(self,
                 in_features,
                 out_features,
                 K=10,
                 alpha=0.1,
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

        lin = []
        lin.append(nn.Dropout(dropout))

        for hid, act in zip(hids, acts):
            lin.append(nn.Linear(in_features,
                                 hid,
                                 bias=bias))
            lin.append(activations.get(act))
            lin.append(nn.Dropout(dropout))
            in_features = hid
        lin = nn.Sequential(*lin)
        conv = SpectralEigenConv(in_features, out_features, bias=bias, K=K, alpha=alpha)

        self.lin = lin
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
        x = self.lin(x)
        x = self.conv(x, U, V)
        return x