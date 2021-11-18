import torch.nn as nn
from torch_geometric.nn import GATConv
from graphgallery.nn.layers.pytorch import Sequential, activations


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
        self.reg_paras = conv[1].parameters()
        self.non_reg_paras = conv[2:].parameters()

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)
