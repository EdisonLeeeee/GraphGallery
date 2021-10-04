import torch.nn as nn
from .gcn import GCNConv


class ChebConv(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 K=3,
                 bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv = nn.ModuleList(GCNConv(in_features, out_features, bias=bias) for _ in range(K))
        self.K = K

    def reset_parameters(self):
        for conv in self.conv:
            conv.reset_parameters()

    def forward(self, x, *adjs):
        out = 0.
        for conv, adj in zip(self.conv, adjs):
            out += conv(x, adj)
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features}, K={self.K})"
