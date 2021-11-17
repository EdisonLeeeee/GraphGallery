import torch.nn as nn
from graphgallery.nn.layers.pytorch import activations


class MLP(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 hids=[16],
                 acts=['relu'],
                 bn=False,
                 dropout=0.5,
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

    def forward(self, x):
        return self.lin(x)
