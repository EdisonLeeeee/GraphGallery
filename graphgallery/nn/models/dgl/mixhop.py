import torch.nn as nn
from torch import optim

from graphgallery.nn.models import TorchEngine
from graphgallery.nn.metrics.pytorch import Accuracy
from graphgallery.nn.layers.pytorch import Sequential, activations
from graphgallery.nn.layers.dgl import MixHopConv


class MixHop(TorchEngine):
    def __init__(self,
                 in_features,
                 out_features, *,
                 hids=[60] * 3,
                 acts=['tanh'] * 3,
                 p=[0, 1, 2],
                 dropout=0.5,
                 weight_decay=5e-4,
                 bias=False,
                 lr=0.1,
                 step_size=40,
                 gamma=0.01):

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
        conv.append(nn.Linear(in_features, out_features, bias=False))
        conv = Sequential(*conv, reverse=True)  # `reverse=True` is important
        self.conv = conv

        optimizer = optim.SGD(self.parameters(),
                              weight_decay=weight_decay, lr=lr)
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optimizer,
                     metrics=[Accuracy()],
                     scheduler=optim.lr_scheduler.StepLR(optimizer, step_size, gamma=gamma))

    def reset_parameters(self):
        for conv in self.conv:
            if hasattr(conv, 'reset_parameters'):
                conv.reset_parameters()

    def forward(self, x, g):
        return self.conv(g, x)
