import torch
import torch.nn.functional as F
from torch import optim
from torch.nn import Module, ModuleList, Dropout

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.layers.pytorch.get_activation import get_activation
from graphgallery.nn.metrics.pytorch import Accuracy

from dgl.nn.pytorch import GraphConv


class GCN(TorchKeras):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hids=[16],
                 acts=['relu'],
                 dropout=0.5,
                 weight_decay=5e-4,
                 lr=0.01,
                 bias=True):

        super().__init__()

        self.layers = ModuleList()

        inc = in_channels
        for hid, act in zip(hids, acts):
            layer = GraphConv(inc,
                              hid,
                              activation=get_activation(act),
                              bias=bias)
            self.layers.append(layer)
            inc = hid
        # output layer
        self.layers.append(GraphConv(inc, out_channels))

        self.dropout = Dropout(p=dropout)
        self.compile(loss=torch.nn.CrossEntropyLoss(),
                     optimizer=optim.Adam(self.parameters(),
                                          lr=lr,
                                          weight_decay=weight_decay),
                     metrics=[Accuracy()])

    def forward(self, x, g):

        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
            x = layer(g, x)

        return x
