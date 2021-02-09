import torch
import torch.nn.functional as F

from torch import optim
from torch import nn

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.layers.pytorch import GraphConvolution
from graphgallery.nn.metrics.pytorch import Accuracy
from graphgallery.nn.layers.pytorch.get_activation import get_activation


class FastGCN(TorchKeras):
    def __init__(self, in_channels, out_channels,
                 hids=[16], acts=['relu'], dropout=0.5,
                 weight_decay=5e-4, lr=0.01, bias=False):

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
            acts_fn.append(get_activation(act))
            paras.append(dict(params=layer.parameters(), weight_decay=weight_decay))
            inc = hid

        conv = GraphConvolution(inc, out_channels, bias=bias)
        # do not use weight_decay in the final layer
        paras.append(dict(params=conv.parameters(), weight_decay=0.))
        self.compile(loss=torch.nn.CrossEntropyLoss(),
                     optimizer=optim.Adam(paras, lr=lr),
                     metrics=[Accuracy()])
        self.dropout = nn.Dropout(dropout)
        self.acts_fn = acts_fn
        self.layers = layers
        self.conv = conv

    def forward(self, x, adj):

        for act, layer in zip(self.acts_fn, self.layers):
            x = act(layer(x))
            x = self.dropout(x)

        return self.conv(x, adj)
