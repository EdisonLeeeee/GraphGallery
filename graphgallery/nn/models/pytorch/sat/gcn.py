import torch.nn as nn
from graphgallery.nn.layers.pytorch import activations, GraphEigenConv, Sequential


class GCN(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 hids=[16],
                 acts=["relu"],
                 dropout=0.5,
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

    def forward(self, x, U, V=None):
        """
        x: feature matrix

        if `V=None`:
            U: (N, N) adjacency matrix
        else:
            U: (N, k) eigenvector matrix
            V: (k,) eigenvalue
        """
        x = self.conv(x, U, V)
        return x
