import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.layers.pytorch import GraphConvolution
from graphgallery.nn.metrics.pytorch import Accuracy
from graphgallery.nn.layers.pytorch.get_activation import get_activation


class MLP(TorchKeras):
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

        layers = nn.ModuleList()
        paras = []
        acts_fn = []

        # use ModuleList to create layers with different size
        inc = in_channels
        for hid, act in zip(hids, acts):
            layer = nn.Linear(inc, hid, bias=use_bias)
            layers.append(layer)
            acts_fn.append(get_activation(act))
            paras.append(dict(params=layer.parameters(), weight_decay=weight_decay))
            inc = hid

        layer = nn.Linear(inc, out_channels, bias=use_bias)
        paras.append(dict(params=layer.parameters(), weight_decay=0.))
        layers.append(layer)
        
        self.layers = layers
        # do not use weight_decay in the final layer
        self.compile(loss=torch.nn.CrossEntropyLoss(),
                     optimizer=optim.Adam(paras, lr=lr),
                     metrics=[Accuracy()])
        self.dropout = nn.Dropout(dropout)
        self.acts_fn = acts_fn

    def forward(self, x):

        for layer, act in zip(self.layers[:-1], self.acts_fn):
            x = self.dropout(x)
            x = act(layer(x))

        x = self.layers[-1](x)
        return x
