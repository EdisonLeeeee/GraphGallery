import torch.nn as nn
from torch import optim

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.metrics.pytorch import Accuracy

from dgl.nn.pytorch.conv import SGConv


class SGC(TorchKeras):
    def __init__(self,
                 in_features,
                 out_features,
                 hids=[],
                 acts=[],
                 K=2,
                 dropout=0.5,
                 weight_decay=5e-5,
                 lr=0.2,
                 bias=True):
        super().__init__()

        if hids or acts:
            raise RuntimeError(
                f"Arguments 'hids' and 'acts' are not supported to use in SGC (DGL backend)."
            )

        conv = SGConv(in_features,
                      out_features,
                      bias=bias,
                      k=K,
                      cached=True)
        self.conv = conv
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam(conv.parameters(),
                                          lr=lr,
                                          weight_decay=weight_decay),
                     metrics=[Accuracy()])

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, g):
        x = self.conv(g, x)
        return x
