import torch.nn as nn
from graphgallery.nn.layers.pytorch import activations
from dgl.nn.pytorch import APPNPConv


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

    def reset_parameters(self):
        for lin in self.lin:
            if hasattr(lin, "reset_parameters"):
                lin.reset_parameters()

    def forward(self, x, g):
        x = self.lin(x)
        x = self.propagation(g, x)
        return x
