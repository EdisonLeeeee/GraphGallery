import torch.nn as nn
from graphgallery.nn.layers.pytorch import activations, EigenConv


class SGC(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 K=2,
                 hids=[],
                 acts=[],
                 dropout=0.5,
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
        conv = EigenConv(in_features, out_features, bias=bias, K=K)

        self.lin = lin
        self.conv = conv

    def forward(self, x, U, V=None):
        """
        x: feature matrix
        if `V=None`:
            U: (N, N) adjacency matrix
        else:
            U: (N, k) eigenvector matrix
            V: (k,) eigenvalue
        """
        x = self.lin(x)
        x = self.conv(x, U, V)
        return x
