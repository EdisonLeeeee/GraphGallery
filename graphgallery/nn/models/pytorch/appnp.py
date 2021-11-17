import torch.nn as nn
from torch import optim

from graphgallery.nn.layers.pytorch import APPNProp, PPNProp, activations


class APPNP(nn.Module):
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

        self.reg_paras = lin[1].parameters()
        self.non_reg_paras = lin[2:].parameters()

    def forward(self, x, adj):
        x = self.lin(x)
        x = self.propagation(x, adj)
        return x
