import torch.nn as nn
from graphgallery.nn.layers.pytorch import (GATConv, SparseGATConv,
                                            Sequential, activations)


class GAT(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 hids=[8],
                 num_heads=[8],
                 acts=['elu'],
                 dropout=0.6,
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

    def forward(self, x, adj):
        return self.conv(x, adj)
