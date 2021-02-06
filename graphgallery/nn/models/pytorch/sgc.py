import torch
import torch.nn.functional as F

from torch.nn import ModuleList, Dropout, Linear
from torch import optim

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.layers.pytorch.get_activation import get_activation
from graphgallery.nn.metrics.pytorch import Accuracy


class SGC(TorchKeras):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hids=[],
                 acts=[],
                 dropout=0.5,
                 weight_decay=5e-5,
                 lr=0.2,
                 use_bias=False):
        super().__init__()

        if len(hids) != len(acts):
            raise RuntimeError(
                f"Arguments 'hids' and 'acts' should have the same length."
                " Or you can set both of them to `[]`.")

        layers = ModuleList()
        acts_fn = []
        paras = []
        inc = in_channels
        for hid, act in zip(hids, acts):
            layer = Linear(inc, hid, bias=use_bias)
            paras.append(
                dict(params=layer.parameters(), weight_decay=weight_decay))
            layers.append(layer)
            inc = hid
            acts_fn.append(get_activation(act))

        layer = Linear(inc, out_channels, bias=use_bias)
        layers.append(layer)
        paras.append(dict(params=layer.parameters(),
                          weight_decay=weight_decay))

        self.layers = layers
        self.acts_fn = acts_fn
        self.dropout = Dropout(dropout)
        self.compile(loss=torch.nn.CrossEntropyLoss(),
                     optimizer=optim.Adam(paras, lr=lr),
                     metrics=[Accuracy()])

    def forward(self, inputs):
        x = inputs

        for layer, act in zip(self.layers[:-1], self.acts_fn):
            x = self.dropout(x)
            x = act(layer(x))

        x = self.layers[-1](x)
        return x
