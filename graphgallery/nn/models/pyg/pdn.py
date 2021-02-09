import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.layers.pytorch.get_activation import get_activation
from graphgallery.nn.metrics.pytorch import Accuracy
from torch_geometric.nn import GCNConv


class PDN(TorchKeras):
    def __init__(self,
                 in_channels,
                 out_channels,
                 edge_channels=1,
                 hids=[32],
                 pdn_hids=32,
                 acts=['relu'],
                 dropout=0.5,
                 weight_decay=5e-4,
                 lr=0.01,
                 bias=True):
        super().__init__()

        convs = nn.ModuleList()
        act_fns = []
        inc = in_channels
        for hid, act in zip(hids, acts):
            layer = GCNConv(inc, hid, bias=bias)
            convs.append(layer)
            act_fns.append(get_activation(act))
            inc = hid
        layer = GCNConv(inc, out_channels, bias=bias)
        convs.append(layer)

        self.fc = nn.Sequential(nn.Linear(edge_channels, pdn_hids),
                                nn.ReLU(),
                                nn.Linear(pdn_hids, 1),
                                nn.Sigmoid())
        self.convs = convs
        self.act_fns = act_fns
        self.dropout = nn.Dropout(dropout)
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam(self.parameters(), lr=lr),
                     metrics=[Accuracy()])

    def forward(self, x, edge_index, edge_x):
        edge_x = self.fc(edge_x).view(-1)

        for layer, act in zip(self.convs, self.act_fns):
            x = act(layer(x, edge_index, edge_x))
            x = self.dropout(x)

        x = self.convs[-1](x, edge_index, edge_x)
        return x
