import torch.nn as nn
from torch import optim

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.layers.pytorch import PropConvolution, activations
from graphgallery.nn.metrics.pytorch import Accuracy


class DAGNN(TorchKeras):
    def __init__(self, in_channels, out_channels, *,
                 hids=[64], acts=['relu'],
                 dropout=0.5, weight_decay=5e-4,
                 lr=0.01, bias=False, K=10):
        super().__init__()

        lin = []
        for hid, act in zip(hids, acts):
            lin.append(nn.Linear(in_channels,
                                 hid,
                                 bias=bias))
            lin.append(activations.get(act))
            lin.append(nn.Dropout(dropout))
            in_channels = hid
        lin.append(nn.Linear(in_channels, out_channels, bias=bias))
        lin.append(activations.get(act))
        lin.append(nn.Dropout(dropout))
        lin = nn.Sequential(*lin)
        self.lin = lin
        self.conv = PropConvolution(out_channels, K=K, bias=bias, activation="sigmoid")
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam([dict(params=lin.parameters(),
                                                weight_decay=weight_decay),
                                           dict(params=self.conv.parameters(), weight_decay=weight_decay),
                                           ], lr=lr),
                     metrics=[Accuracy()])

    def forward(self, x, adj):
        x = self.lin(x)
        return self.conv(x, adj)
