import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.layers.pytorch import PropConvolution
from graphgallery.nn.metrics.pytorch import Accuracy
from graphgallery.nn.layers.pytorch.get_activation import get_activation


class DAGNN(TorchKeras):
    def __init__(self, in_channels, out_channels,
                 hids=[64], acts=['relu'],
                 dropout=0.5, weight_decay=5e-3,
                 lr=0.01, bias=False, K=10):
        super().__init__()

        layers = nn.ModuleList()
        acts_fn = []

        # use ModuleList to create layers with different size
        inc = in_channels
        for hid, act in zip(hids, acts):
            layer = nn.Linear(inc, hid, bias=bias)
            layers.append(layer)
            acts_fn.append(get_activation(act))
            inc = hid

        layer = nn.Linear(inc, out_channels, bias=bias)
        acts_fn.append(get_activation(act))
        layers.append(layer)

        conv = PropConvolution(out_channels, K=K, bias=bias, activation="sigmoid")
        self.layers = layers
        self.conv = conv
        paras = [dict(params=layers.parameters(), weight_decay=weight_decay),
                 dict(params=conv.parameters(), weight_decay=weight_decay),
                 ]

        # do not use weight_decay in the final layer
        self.compile(loss=torch.nn.CrossEntropyLoss(),
                     optimizer=optim.Adam(paras, lr=lr),
                     metrics=[Accuracy()])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):

        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)

        return self.conv(x, adj)
