import torch.nn as nn
from torch import optim

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.layers.pytorch import Sequential, activations
from graphgallery.nn.metrics.pytorch import Accuracy

from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_adj


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
            conv.append(GCNConv(in_features,
                                hid,
                                cached=True,
                                bias=bias,
                                normalize=False))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_features = hid
        conv.append(GCNConv(in_features,
                            out_features,
                            cached=True,
                            bias=bias,
                            normalize=False))
        conv = Sequential(*conv)

        self.conv = conv
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam([dict(params=conv[0].parameters(),
                                                weight_decay=weight_decay),
                                           dict(params=conv[1:].parameters(),
                                                weight_decay=0.)], lr=lr),
                     metrics=[Accuracy()])

    def forward(self, x, edge_index, edge_weight=None):
        return self.conv(x, edge_index, edge_weight)

class DropEdge(TorchKeras):
    def __init__(self,
                 in_features,
                 out_features,
                 p=0.05,
                 hids=[16],
                 acts=['relu'],
                 dropout=0.5,
                 weight_decay=5e-4,
                 lr=0.01,
                 bias=True):

        super().__init__()

        conv = []
        conv.append(nn.Dropout(dropout))
        for hid, act in zip(hids, acts):
            conv.append(GCNConv(in_features,
                                hid,
                                cached=True,
                                bias=bias,
                                normalize=False))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_features = hid
        conv.append(GCNConv(in_features,
                            out_features,
                            cached=True,
                            bias=bias,
                            normalize=False))
        conv = Sequential(*conv)

        self.p = p
        self.conv = conv
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam([dict(params=conv[1].parameters(),
                                                weight_decay=weight_decay),
                                           dict(params=conv[2:].parameters(),
                                                weight_decay=0.)], lr=lr),
                     metrics=[Accuracy()])

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, edge_weight = dropout_adj(edge_index, edge_weight, p=self.p)
        return self.conv(x, edge_index, edge_weight)