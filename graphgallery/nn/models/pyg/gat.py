import torch
import torch.nn.functional as F

from torch.nn import Module, ModuleList, Dropout
from torch import optim
from torch_geometric.nn import GATConv

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.layers.pytorch.get_activation import get_activation
from graphgallery.nn.metrics.pytorch import Accuracy


class GAT(TorchKeras):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hids=[8],
                 num_heads=[8],
                 acts=['elu'],
                 dropout=0.6,
                 weight_decay=5e-4,
                 lr=0.01,
                 use_bias=True):

        super().__init__()

        layers = ModuleList()
        act_fns = []
        paras = []

        inc = in_channels
        pre_head = 1
        for hid, num_head, act in zip(hids, num_heads, acts):
            layer = GATConv(inc * pre_head,
                            hid,
                            heads=num_head,
                            bias=use_bias,
                            dropout=dropout)
            layers.append(layer)
            act_fns.append(get_activation(act))
            paras.append(
                dict(params=layer.parameters(), weight_decay=weight_decay))
            inc = hid
            pre_head = num_head

        layer = GATConv(inc * pre_head,
                        out_channels,
                        heads=1,
                        bias=use_bias,
                        concat=False,
                        dropout=dropout)
        layers.append(layer)
        # do not use weight_decay in the final layer
        paras.append(dict(params=layer.parameters(), weight_decay=0.))

        self.act_fns = act_fns
        self.layers = layers
        self.dropout = Dropout(dropout)
        self.compile(loss=torch.nn.CrossEntropyLoss(),
                     optimizer=optim.Adam(paras, lr=lr),
                     metrics=[Accuracy()])

    def forward(self, x, edge_index, edge_weight=None):
        x = self.dropout(x)

        for layer, act in zip(self.layers, self.act_fns):
            x = act(layer(x, edge_index))
            x = self.dropout(x)

        x = self.layers[-1](x, edge_index)
        return x
