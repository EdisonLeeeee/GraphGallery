import torch.nn as nn
from torch import optim

from graphgallery.nn.models import TorchEngine
from graphgallery.nn.layers.pytorch import GATConv, SparseGATConv, Sequential, activations
from graphgallery.nn.metrics.pytorch import Accuracy


class GAT(TorchEngine):
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
            conv.append(SparseGATConv(in_features * head,
                                      hid,
                                      attn_heads=num_head,
                                      reduction='concat',
                                      bias=bias))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_features = hid
            head = num_head
        conv.append(SparseGATConv(in_features * head,
                                  out_features,
                                  attn_heads=1,
                                  reduction='average',
                                  bias=bias))
        conv = Sequential(*conv)

        self.conv = conv
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam([dict(params=conv[1].parameters(),
                                                weight_decay=weight_decay),
                                           dict(params=conv[2:].parameters(),
                                                weight_decay=0.)], lr=lr),
                     metrics=[Accuracy()])

    def forward(self, x, adj):
        return self.conv(x, adj)
