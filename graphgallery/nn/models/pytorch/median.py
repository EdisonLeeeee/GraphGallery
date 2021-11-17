import torch.nn as nn
from graphgallery.nn.layers.pytorch import TrimmedConv, MedianConv, Sequential, activations


class MedianGCN(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 hids=[16],
                 acts=['relu'],
                 dropout=0.5,
                 bias=False):

        super().__init__()
        conv = []
        conv.append(nn.Dropout(dropout))
        for hid, act in zip(hids, acts):
            conv.append(MedianConv(in_features,
                                   hid,
                                   bias=bias))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_features = hid
        conv.append(MedianConv(in_features, out_features, bias=bias))
        conv = Sequential(*conv)

        self.conv = conv
        self.reg_paras = conv[1].parameters()
        self.non_reg_paras = conv[2:].parameters()

    def forward(self, x, adj):
        return self.conv(x, adj)


class TrimmedGCN(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 hids=[16],
                 acts=['relu'],
                 tperc=0.45,
                 dropout=0.5,
                 bias=False):

        super().__init__()
        conv = []
        conv.append(nn.Dropout(dropout))
        for hid, act in zip(hids, acts):
            conv.append(TrimmedConv(in_features,
                                    hid,
                                    bias=bias,
                                    tperc=tperc))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_features = hid
        conv.append(TrimmedConv(in_features, out_features,
                                bias=bias,
                                tperc=tperc))
        conv = Sequential(*conv)

        self.conv = conv
        self.reg_paras = conv[1].parameters()
        self.non_reg_paras = conv[2:].parameters()

    def forward(self, x, adj):
        return self.conv(x, adj)
