import torch.nn as nn
from torch import optim

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.layers.pytorch import Sequential, activations, AGNNConv
from graphgallery.nn.metrics.pytorch import Accuracy


class AGNN(TorchKeras):

    def __init__(self,
                 in_features,
                 out_features,
                 *,
                 hids=[16],
                 num_attn=2,
                 acts=['relu'],
                 dropout=0.5,
                 weight_decay=5e-4,
                 lr=0.01,
                 bias=False):
        super().__init__()
        conv = []

        for hid, act in zip(hids, acts):
            conv.append(nn.Linear(in_features, hid, bias=bias))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_features = hid

        # for Cora dataset, the first propagation layer is non-trainable
        # and beta is fixed at 0
        conv.append(AGNNConv(trainable=False))
        for _ in range(1, num_attn):
            conv.append(AGNNConv())

        conv.append(nn.Linear(in_features, out_features, bias=bias))
        conv.append(nn.Dropout(dropout))
        conv = Sequential(*conv)
        self.conv = conv
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam([dict(params=conv[0].parameters(),
                                                weight_decay=weight_decay),
                                           dict(params=conv[1:].parameters(),
                                                weight_decay=0.)], lr=lr),
                     metrics=[Accuracy()])

    def forward(self, x, adj):
        return self.conv(x, adj)
