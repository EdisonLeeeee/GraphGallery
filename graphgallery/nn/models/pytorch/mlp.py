import torch.nn as nn
from torch import optim

from graphgallery.nn.models import TorchEngine
from graphgallery.nn.metrics.pytorch import Accuracy
from graphgallery.nn.layers.pytorch import activations


class MLP(TorchEngine):
    def __init__(self,
                 in_features,
                 out_features,
                 hids=[16],
                 acts=['relu'],
                 bn=False,
                 dropout=0.5,
                 weight_decay=5e-4,
                 lr=0.01,
                 bias=False):

        super().__init__()

        lin = []
        lin.append(nn.Dropout(dropout))

        for hid, act in zip(hids, acts):
            if bn:
                lin.append(nn.BatchNorm1d(in_features))
            lin.append(nn.Linear(in_features,
                                 hid,
                                 bias=bias))
            lin.append(activations.get(act))
            lin.append(nn.Dropout(dropout))
            in_features = hid
        if bn:
            lin.append(nn.BatchNorm1d(in_features))
        lin.append(nn.Linear(in_features, out_features, bias=bias))
        lin = nn.Sequential(*lin)

        self.lin = lin
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam(self.parameters(),
                                          weight_decay=weight_decay, lr=lr),
                     metrics=[Accuracy()])

    def forward(self, x):
        return self.lin(x)
