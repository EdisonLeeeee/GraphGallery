import torch
import torch.nn as nn
from torch import optim

from graphgallery.nn.models import TorchEngine
from graphgallery.nn.layers.pytorch import GCNConv, Sequential, activations
from graphgallery.nn.metrics.pytorch import Accuracy
# from torch_geometric.utils import dropout_adj


class GCN(TorchEngine):
    def __init__(self,
                 in_features,
                 out_features,
                 *,
                 hids=[16],
                 acts=['relu'],
                 dropout=0.5,
                 weight_decay=5e-4,
                 lr=0.01,
                 bias=False):
        super().__init__()
        conv = []
        conv.append(nn.Dropout(dropout))
        for hid, act in zip(hids, acts):
            conv.append(GCNConv(in_features,
                                hid,
                                bias=bias))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_features = hid
        conv.append(GCNConv(in_features, out_features, bias=bias))
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


# class DropEdge(TorchEngine):
#     def __init__(self,
#                  in_features,
#                  out_features,
#                  *,
#                  p=0.05,
#                  hids=[16],
#                  acts=['relu'],
#                  dropout=0.5,
#                  weight_decay=5e-4,
#                  lr=0.01,
#                  bias=False):
#         super().__init__()
#         conv = []
#         conv.append(nn.Dropout(dropout))
#         for hid, act in zip(hids, acts):
#             conv.append(GCNConv(in_features,
#                                 hid,
#                                 bias=bias))
#             conv.append(activations.get(act))
#             conv.append(nn.Dropout(dropout))
#             in_features = hid
#         conv.append(GCNConv(in_features, out_features, bias=bias))
#         conv = Sequential(*conv)

#         self.p = p
#         self.conv = conv
#         self.compile(loss=nn.CrossEntropyLoss(),
#                      optimizer=optim.Adam([dict(params=conv[1].parameters(),
#                                                 weight_decay=weight_decay),
#                                            dict(params=conv[2:].parameters(),
#                                                 weight_decay=0.)], lr=lr),
#                      metrics=[Accuracy()])

#     def forward(self, x, adj):
#         if self.training:
#             adj = adj.coalesce()
#             edge_index = adj.indices()
#             edge_weight = adj.values()
#             edge_index, edge_weight = dropout_adj(edge_index, edge_weight, p=self.p, force_undirected=True)
#             adj = torch.sparse.FloatTensor(edge_index, edge_weight, adj.size())

#         return self.conv(x, adj)
