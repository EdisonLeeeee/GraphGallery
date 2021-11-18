import torch.nn as nn

from graphgallery.nn.layers.pytorch import Sequential, activations
from graphgallery.nn.layers.pyg import MedianConv


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
                                   add_self_loops=False,
                                   bias=bias))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_features = hid
        conv.append(MedianConv(in_features,
                               out_features,
                               add_self_loops=False,
                               bias=bias))
        conv = Sequential(*conv)

        self.conv = conv
        self.reg_paras = conv[0].parameters()
        self.non_reg_paras = conv[1:].parameters()

    def forward(self, x, edge_index, edge_weight=None):
        return self.conv(x, edge_index, edge_weight)
