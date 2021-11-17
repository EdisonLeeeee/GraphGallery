import torch.nn as nn
from graphgallery.nn.layers.dgl import LGConv, EGConv, hLGConv


class LGC(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 hids=[],
                 acts=[],
                 K=20,
                 dropout=0.0,
                 bias=True):
        super().__init__()

        if hids or acts:
            raise RuntimeError(
                f"Arguments 'hids' and 'acts' are not supported to use in {self.__class__.__name__} (DGL backend).")

        conv = LGConv(in_features,
                      out_features,
                      bias=bias,
                      k=K,
                      cached=True)
        self.dropout = nn.Dropout(dropout)
        self.conv = conv

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, g):
        x = self.dropout(x)
        x = self.conv(g, x)
        return x


class EGC(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 hids=[],
                 acts=[],
                 K=20,
                 dropout=0.0,
                 bias=True):
        super().__init__()

        if hids or acts:
            raise RuntimeError(
                f"Arguments 'hids' and 'acts' are not supported to use in {self.__class__.__name__} (DGL backend).")

        conv = EGConv(in_features,
                      out_features,
                      bias=bias,
                      k=K,
                      cached=True)
        self.dropout = nn.Dropout(dropout)
        self.conv = conv

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, g):
        x = self.dropout(x)
        x = self.conv(g, x)
        return x


class hLGC(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 hids=[],
                 acts=[],
                 K=20,
                 dropout=0.0,
                 bias=True):
        super().__init__()

        if hids or acts:
            raise RuntimeError(
                f"Arguments 'hids' and 'acts' are not supported to use in {self.__class__.__name__} (DGL backend).")

        conv = hLGConv(in_features,
                       out_features,
                       bias=bias,
                       k=K,
                       cached=True)
        self.dropout = nn.Dropout(dropout)
        self.conv = conv

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, g):
        x = self.dropout(x)
        x = self.conv(g, x)
        return x
