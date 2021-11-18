import torch.nn as nn
from graphgallery.nn.layers.pytorch import Sequential, activations
from torch_geometric.nn import GCNConv


class PDN(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 in_edge_features,
                 *,
                 hids=[32],
                 pdn_hids=32,
                 acts=['relu'],
                 dropout=0.5,
                 bias=True):
        super().__init__()

        conv = []
        for hid, act in zip(hids, acts):
            conv.append(GCNConv(in_features,
                                hid,
                                bias=bias))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_features = hid

        conv.append(GCNConv(in_features,
                            out_features,
                            bias=bias))
        conv = Sequential(*conv)

        self.fc = nn.Sequential(nn.Linear(in_edge_features, pdn_hids),
                                nn.ReLU(),
                                nn.Linear(pdn_hids, 1),
                                nn.Sigmoid())
        self.conv = conv

    def forward(self, x, edge_index, edge_x):
        edge_x = self.fc(edge_x).view(-1)
        x = self.conv(x, edge_index, edge_x)
        return x
