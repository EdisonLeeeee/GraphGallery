import torch.nn as nn
from torch import optim

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.layers.pytorch import TAGConv, Sequential, activations
from graphgallery.nn.metrics.pytorch import Accuracy


class TAGCN(TorchKeras):
    def __init__(self,
                 in_features,
                 out_features,
                 hids=[16],
                 acts=['relu'],
                 K=3,
                 dropout=0.5,
                 weight_decay=5e-4,
                 lr=0.01,
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
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam([dict(params=conv[1].parameters(),
                                                weight_decay=weight_decay),
                                           dict(params=conv[2:].parameters(),
                                                weight_decay=0.)], lr=lr),
                     metrics=[Accuracy()])

    def forward(self, x, adj):
        return self.conv(x, adj)
