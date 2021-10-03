import torch.nn as nn
from torch import optim

from graphgallery.nn.models import TorchEngine
from graphgallery.nn.metrics.pytorch import Accuracy
from graphgallery.nn.layers.dgl import LGConv, EGConv, hLGConv


class LGC(TorchEngine):
    def __init__(self,
                 in_features,
                 out_features,
                 hids=[],
                 acts=[],
                 K=20,
                 dropout=0.0,
                 weight_decay=5e-5,
                 lr=0.2,
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
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam(conv.parameters(),
                                          lr=lr,
                                          weight_decay=weight_decay),
                     metrics=[Accuracy()])

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, g):
        x = self.dropout(x)
        x = self.conv(g, x)
        return x


class EGC(TorchEngine):
    def __init__(self,
                 in_features,
                 out_features,
                 hids=[],
                 acts=[],
                 K=20,
                 dropout=0.0,
                 weight_decay=5e-5,
                 lr=0.2,
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
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam(conv.parameters(),
                                          lr=lr,
                                          weight_decay=weight_decay),
                     metrics=[Accuracy()])

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, g):
        x = self.dropout(x)
        x = self.conv(g, x)
        return x


class hLGC(TorchEngine):
    def __init__(self,
                 in_features,
                 out_features,
                 hids=[],
                 acts=[],
                 K=20,
                 dropout=0.0,
                 weight_decay=5e-5,
                 lr=0.2,
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
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam(conv.parameters(),
                                          lr=lr,
                                          weight_decay=weight_decay),
                     metrics=[Accuracy()])

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, g):
        x = self.dropout(x)
        x = self.conv(g, x)
        return x
