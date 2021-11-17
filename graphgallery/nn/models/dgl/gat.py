import torch.nn as nn
from torch import optim

from graphgallery.nn.models import TorchEngine
from graphgallery.nn.layers.pytorch import Sequential, activations
from graphgallery.nn.metrics import Accuracy

from dgl.nn.pytorch import GATConv


class GAT(TorchEngine):
    def __init__(self,
                 in_features,
                 out_features,
                 hids=[8],
                 num_heads=[8],
                 acts=['elu'],
                 dropout=0.6,
                 bias=False,
                 weight_decay=5e-4,
                 lr=0.01):

        super().__init__()
        head = 1
        conv = []
        for hid, num_head, act in zip(hids, num_heads, acts):
            conv.append(GATConv(in_features * head,
                                hid,
                                bias=bias,
                                num_heads=num_head,
                                feat_drop=dropout,
                                attn_drop=dropout))
            conv.append(activations.get(act))
            conv.append(nn.Flatten(start_dim=1))
            conv.append(nn.Dropout(dropout))
            in_features = hid
            head = num_head

        conv.append(GATConv(in_features * head,
                            out_features,
                            num_heads=1,
                            bias=bias,
                            feat_drop=dropout,
                            attn_drop=dropout))
        conv = Sequential(*conv, reverse=True)  # `reverse=True` is important

        self.conv = conv
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam([dict(params=conv[0].parameters(),
                                                weight_decay=weight_decay),
                                           dict(params=conv[1:].parameters(),
                                                weight_decay=0.)], lr=lr),
                     metrics=[Accuracy()])

    def reset_parameters(self):
        for conv in self.conv:
            conv.reset_parameters()

    def forward(self, x, g):
        x = self.conv(g, x).mean(1)
        return x
