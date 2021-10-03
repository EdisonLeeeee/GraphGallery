import torch.nn as nn
from torch import optim

from graphgallery.nn.models import TorchEngine
from graphgallery.nn.layers.pytorch import APPNProp, PPNProp, activations
from graphgallery.nn.metrics.pytorch import Accuracy


class APPNP(TorchEngine):
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
                 bias=True,
                 approximated=True):

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
        if approximated:
            self.propagation = APPNProp(alpha=alpha, K=K,
                                        dropout=ppr_dropout)
        else:
            self.propagation = PPNProp(dropout=ppr_dropout)
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam([dict(params=lin[1].parameters(),
                                                weight_decay=weight_decay),
                                           dict(params=lin[2:].parameters(),
                                                weight_decay=0.)], lr=lr),
                     metrics=[Accuracy()])

    def reset_parameters(self):
        for layer in self.lin:
            layer.reset_parameters()

    def forward(self, x, adj):
        x = self.lin(x)
        x = self.propagation(x, adj)
        return x
