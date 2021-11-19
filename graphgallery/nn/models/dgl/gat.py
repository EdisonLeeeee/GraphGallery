import torch.nn as nn
from graphgallery.nn.layers.pytorch import Sequential, activations
from dgl.nn.pytorch import GATConv


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
        conv = Sequential(*conv, loc=1)  # loc=1 specifies the location of features

        self.conv = conv

    def reset_parameters(self):
        for conv in self.conv:
            conv.reset_parameters()

    def forward(self, x, g):
        return self.conv(g, x).mean(1)
