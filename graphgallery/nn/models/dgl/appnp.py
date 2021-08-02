import torch.nn as nn
from torch import optim

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.metrics.pytorch import Accuracy
from graphgallery.nn.layers.pytorch import Sequential, activations

from dgl.nn.pytorch import APPNPConv


class APPNP(TorchKeras):
    def __init__(self,
                 in_features,
                 out_features,
                 *,
                 alpha=0.1,
                 K=10,
                 ppr_dropout=0.,
                 hids=[64],
                 acts=['relu'],
                 dropout=0.5,
                 weight_decay=5e-4,
                 lr=0.01,
                 bias=True):

        super().__init__()

        lin = []
        lin.append(nn.Dropout(dropout))
        for hid, act in zip(hids, acts):
            lin.append(nn.Linear(in_features,
                                 hid,
                                 bias=bias))
            lin.append(activations.get(act))
            lin.append(nn.Dropout(dropout))
            in_features = hid
        lin.append(nn.Linear(in_features, out_features, bias=bias))
        lin = nn.Sequential(*lin)
        self.lin = lin
        self.propagation = APPNPConv(K, alpha, ppr_dropout)
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam(self.parameters(),
                                          weight_decay=weight_decay, lr=lr),
                     metrics=[Accuracy()])

    def reset_parameters(self):
        for lin in self.lin:
            lin.reset_parameters()

    def forward(self, x, g):
        x = self.lin(x)
        x = self.propagation(g, x)
        return x
