import torch.nn as nn
from torch import optim

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.metrics.pytorch import Accuracy
from graphgallery.nn.layers.pytorch import Sequential, activations

from dgl.nn.pytorch import GraphConv


class GCN(TorchKeras):
    def __init__(self,
                 in_features,
                 out_features,
                 hids=[16],
                 acts=['relu'],
                 dropout=0.5,
                 weight_decay=5e-4,
                 lr=0.01,
                 bias=True):

        super().__init__()

        conv = []
        for hid, act in zip(hids, acts):
            conv.append(GraphConv(in_features,
                                  hid,
                                  bias=bias))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_features = hid
        conv.append(GraphConv(in_features, out_features))
        conv = Sequential(*conv, inverse=True)  # `inverse=True` is important

        self.conv = conv
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam([dict(params=conv[0].parameters(),
                                                weight_decay=weight_decay),
                                           dict(params=conv[1:].parameters(),
                                                weight_decay=0.)], lr=lr),
                     metrics=[Accuracy()])

    def forward(self, x, g):
        return self.conv(g, x)
