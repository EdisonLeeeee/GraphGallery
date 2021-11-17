import torch.nn as nn
from torch import optim
from torch_geometric.nn import GATConv

from graphgallery.nn.layers.pytorch import Sequential, activations
from graphgallery.nn.metrics import Accuracy


class GAT(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 hids=[8],
                 num_heads=[8],
                 acts=['elu'],
                 dropout=0.6,
                 weight_decay=5e-4,
                 lr=0.01,
                 bias=True):

        super().__init__()
        head = 1
        conv = []
        conv.append(nn.Dropout(dropout))
        for hid, num_head, act in zip(hids, num_heads, acts):
            conv.append(GATConv(in_features * head,
                                hid,
                                heads=num_head,
                                bias=bias,
                                dropout=dropout))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_features = hid
            head = num_head

        conv.append(GATConv(in_features * head,
                            out_features,
                            heads=1,
                            bias=bias,
                            concat=False,
                            dropout=dropout))
        conv = Sequential(*conv)

        self.conv = conv
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam([dict(params=conv[1].parameters(),
                                                weight_decay=weight_decay),
                                           dict(params=conv[2:].parameters(),
                                                weight_decay=0.)], lr=lr),
                     metrics=[Accuracy()])

    def forward(self, x, edge_index, edge_weight=None):
        return self.conv(x, edge_index)
