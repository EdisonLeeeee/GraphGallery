import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.layers.pytorch import TAGConvolution
from graphgallery.nn.metrics.pytorch import Accuracy


class TAGCN(TorchKeras):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hids=[16],
                 acts=['relu'],
                 K=3,
                 dropout=0.5,
                 weight_decay=5e-4,
                 lr=0.01,
                 use_bias=False):

        super().__init__()

        layers = nn.ModuleList()

        # use ModuleList to create layers with different size
        inc = in_channels
        for hid, act in zip(hids, acts):
            layer = TAGConvolution(inc,
                                   hid, K=K,
                                   activation=act,
                                   use_bias=use_bias)
            layers.append(layer)
            inc = hid

        layer = TAGConvolution(inc, out_channels, K=K,
                               use_bias=use_bias)
        layers.append(layer)
        self.layers = layers
        self.compile(loss=torch.nn.CrossEntropyLoss(),
                     optimizer=optim.Adam(layers.parameters(),
                                          weight_decay=weight_decay, lr=lr),
                     metrics=[Accuracy()])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        for layer in self.layers:
            x = self.dropout(x)
            x = layer(x, adj)
        return x
