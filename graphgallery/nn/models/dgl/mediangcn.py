import torch.nn as nn
from graphgallery.nn.layers.pytorch import Sequential, activations
from graphgallery.nn.layers.dgl import MedianConv


class MedianGCN(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 hids=[16],
                 acts=['relu'],
                 dropout=0.5,
                 bias=True):

        super().__init__()

        conv = []
        for hid, act in zip(hids, acts):
            conv.append(MedianConv(in_features,
                                   hid,
                                   bias=bias,
                                   activation=activations.get(act)))
            conv.append(nn.Dropout(dropout))
            in_features = hid
        conv.append(MedianConv(in_features, out_features))
        conv = Sequential(*conv, loc=1)  # loc=1 specifies the location of features

        self.conv = conv

    def reset_parameters(self):
        for conv in self.conv:
            if hasattr(conv, 'reset_parameters'):
                conv.reset_parameters()

    def forward(self, x, g):
        return self.conv(g, x)
