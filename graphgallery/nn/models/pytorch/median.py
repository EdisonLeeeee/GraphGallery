import torch
import torch.nn.functional as F

from torch.nn import Module, ModuleList, Dropout
from torch import optim

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.layers.pytorch import MedianConvolution
from graphgallery.nn.metrics.pytorch import Accuracy


class MedianGCN(TorchKeras):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hids=[16],
                 acts=['relu'],
                 dropout=0.5,
                 weight_decay=5e-4,
                 lr=0.01,
                 use_bias=False):

        super().__init__()

        layers = ModuleList()
        paras = []

        # use ModuleList to create layers with different size
        inc = in_channels
        for hid, act in zip(hids, acts):
            layer = MedianConvolution(inc,
                                      hid,
                                      activation=act,
                                      use_bias=use_bias)
            layers.append(layer)
            paras.append(dict(params=layer.parameters(), weight_decay=weight_decay))
            inc = hid

        layer = MedianConvolution(inc, out_channels, use_bias=use_bias)
        layers.append(layer)
        # do not use weight_decay in the final layer
        paras.append(dict(params=layer.parameters(), weight_decay=0.))
        self.compile(loss=torch.nn.CrossEntropyLoss(),
                     optimizer=optim.Adam(paras, lr=lr),
                     metrics=[Accuracy()])
        self.dropout = Dropout(dropout)
        self.layers = layers

    def forward(self, x, nbrs):

        for layer in self.layers:
            x = self.dropout(x)
            x = layer(x, nbrs)

        return x
