import torch.nn as nn
from torch import optim

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.layers.pytorch import GraphConvolution, Sequential, activations
from graphgallery.nn.metrics.pytorch import Accuracy


class GCN(TorchKeras):
    def __init__(self,
                 in_channels,
                 out_channels,
                 *,
                 hids=[16],
                 acts=['relu'],
                 dropout=0.5,
                 weight_decay=5e-4,
                 lr=0.01,
                 bias=False):
        super().__init__()
        conv = []
        conv.append(nn.Dropout(dropout))
        inc = in_channels
        for hid, act in zip(hids, acts):
            conv.append(GraphConvolution(inc,
                                         hid,
                                         bias=bias))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            inc = hid
        conv.append(GraphConvolution(inc, out_channels, bias=bias))
        conv = Sequential(*conv)

        self.conv = conv
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam([dict(params=conv[1].parameters(),
                                                weight_decay=weight_decay),
                                           dict(params=conv[2:].parameters(),
                                                weight_decay=0.), ], lr=lr),
                     metrics=[Accuracy()])

    def forward(self, x, adj):
        return self.conv(x, adj)
