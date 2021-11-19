import torch.nn as nn
from graphgallery.nn.layers.pytorch import Sequential, activations
from graphgallery.nn.layers.dgl import MixHopConv


class MixHop(nn.Module):
    def __init__(self,
                 in_features,
                 out_features, *,
                 hids=[60] * 3,
                 acts=['tanh'] * 3,
                 p=[0, 1, 2],
                 dropout=0.5,
                 bias=False):

        super().__init__()
        conv = []
        conv.append(nn.Dropout(dropout))

        for hid, act in zip(hids, acts):
            conv.append(MixHopConv(in_features,
                                   hid,
                                   p=p,
                                   bias=bias))
            conv.append(nn.BatchNorm1d(hid * len(p)))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_features = hid * len(p)
        conv.append(nn.Linear(in_features, out_features, bias=bias))
        conv = Sequential(*conv, loc=1)  # loc=1 specifies the location of features
        self.conv = conv

    def reset_parameters(self):
        for conv in self.conv:
            if hasattr(conv, 'reset_parameters'):
                conv.reset_parameters()

    def forward(self, x, g):
        return self.conv(g, x)
