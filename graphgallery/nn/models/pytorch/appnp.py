import torch
import torch.nn as nn
from torch import optim

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.layers.pytorch import APPNPropagation, PPNPropagation, MixedDropout
from graphgallery.nn.metrics.pytorch import Accuracy
from graphgallery.nn.layers.pytorch.get_activation import get_activation


class APPNP(TorchKeras):
    def __init__(self,
                 in_channels,
                 out_channels,
                 alpha=0.1,
                 K=10,
                 ppr_dropout=0.,
                 hids=[64],
                 acts=['relu'],
                 dropout=0.5,
                 weight_decay=5e-4,
                 lr=0.01,
                 bias=True,
                 approximated=True):

        super().__init__()

        layers = nn.ModuleList()
        paras = []
        acts_fn = []

        # use ModuleList to create layers with different size
        inc = in_channels
        for hid, act in zip(hids, acts):
            layer = nn.Linear(inc,
                              hid,
                              bias=bias)
            layers.append(layer)
            paras.append(dict(params=layer.parameters(), weight_decay=weight_decay))
            acts_fn.append(get_activation(act))
            inc = hid

        layer = nn.Linear(inc, out_channels, bias=bias)
        paras.append(dict(params=layer.parameters(), weight_decay=0.))
        layers.append(layer)
        self.layers = layers
        self.acts_fn = acts_fn
        self.dropout = MixedDropout(dropout)
        if approximated:
            self.propagation = APPNPropagation(alpha=alpha, K=K,
                                               dropout=ppr_dropout)
        else:
            self.propagation = PPNPropagation(dropout=ppr_dropout)
        # do not use weight_decay in the final layer
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam(paras, lr=lr),
                     metrics=[Accuracy()])
        self.act_fn = nn.ReLU()

    def forward(self, x, adj):
        for layer, act in zip(self.layers[:-1], self.acts_fn):
            x = self.dropout(x)
            x = act(layer(x))
        x = self.layers[-1](self.dropout(x))
        x = self.propagation(x, adj)
        return x
