import torch.nn as nn
from graphgallery.nn.layers.dgl import DAGNNConv
from graphgallery.nn.layers.pytorch import activations


class DAGNN(nn.Module):
    def __init__(self, in_features, out_features,
                 *,
                 hids=[64],
                 acts=['relu'],
                 dropout=0.5,
                 bias=False,
                 K=10):
        super().__init__()

        lin = []
        for hid, act in zip(hids, acts):
            lin.append(nn.Dropout(dropout))
            lin.append(nn.Linear(in_features,
                                 hid,
                                 bias=bias))
            lin.append(activations.get(act))
            in_features = hid
        lin.append(nn.Linear(in_features, out_features, bias=bias))
        lin.append(activations.get(act))
        lin.append(nn.Dropout(dropout))
        lin = nn.Sequential(*lin)
        self.lin = lin
        self.conv = DAGNNConv(out_features, K=K, bias=bias)

    def reset_parameters(self):
        for lin in self.lin:
            if hasattr(lin, "reset_parameters"):
                lin.reset_parameters()
        self.conv.reset_parameters()

    def forward(self, x, g):
        x = self.lin(x)
        return self.conv(g, x)
