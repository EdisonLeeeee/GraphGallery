import torch
import torch.nn.functional as F
from torch import optim
from torch.nn import Module, ModuleList, Dropout

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.metrics.pytorch import Accuracy

from dgl.nn.pytorch.conv import SGConv


class SGC(TorchKeras):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hids=[],
                 acts=[],
                 K=2,
                 dropout=0.5,
                 weight_decay=5e-5,
                 lr=0.2,
                 use_bias=True):
        super().__init__()

        if hids or acts:
            raise RuntimeError(
                f"Arguments 'hids' and 'acts' are not supported to use in SGC (DGL backend)."
            )

        conv = SGConv(in_channels,
                      out_channels,
                      bias=use_bias,
                      k=K,
                      cached=True)
        self.conv = conv
        self.dropout = Dropout(dropout)
        self.compile(loss=torch.nn.CrossEntropyLoss(),
                     optimizer=optim.Adam(conv.parameters(),
                                          lr=lr,
                                          weight_decay=weight_decay),
                     metrics=[Accuracy()])

    def forward(self, x, g):
        x = self.conv(g, x)
        return x
