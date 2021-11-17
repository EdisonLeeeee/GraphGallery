import torch.nn as nn

from graphgallery.nn.layers.pytorch import TAGConv, Sequential, activations


class TAGCN(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 hids=[16],
                 acts=['relu'],
                 K=3,
                 dropout=0.5,
                 bias=False):
        super().__init__()
        conv = []
        conv.append(nn.Dropout(dropout))
        for hid, act in zip(hids, acts):
            conv.append(TAGConv(in_features,
                                hid, K=K,
                                bias=bias))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_features = hid
        conv.append(TAGConv(in_features,
                            out_features, K=K,
                            bias=bias))
        conv = Sequential(*conv)

        self.conv = conv

    def forward(self, x, adj):
        return self.conv(x, adj)
