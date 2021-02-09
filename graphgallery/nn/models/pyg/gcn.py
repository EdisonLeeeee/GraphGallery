import torch
import torch.nn.functional as F
from torch import optim
from torch.nn import Module, ModuleList, Dropout

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.layers.pytorch.get_activation import get_activation
from graphgallery.nn.metrics.pytorch import Accuracy

from torch_geometric.nn import GCNConv


class GCN(TorchKeras):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hids=[16],
                 acts=['relu'],
                 dropout=0.5,
                 weight_decay=5e-4,
                 lr=0.01,
                 bias=True):

        super().__init__()

        paras = []
        act_fns = []
        layers = ModuleList()
        # use ModuleList to create layers with different size
        inc = in_channels
        for hid, act in zip(hids, acts):
            layer = GCNConv(inc,
                            hid,
                            cached=True,
                            bias=bias,
                            normalize=False)
            layers.append(layer)
            paras.append(
                dict(params=layer.parameters(), weight_decay=weight_decay))
            act_fns.append(get_activation(act))
            inc = hid

        layer = GCNConv(inc,
                        out_channels,
                        cached=True,
                        bias=bias,
                        normalize=False)
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

        for layer, act in zip(self.layers, self.act_fns):
            x = act(layer(x, edge_index, edge_weight))
            x = self.dropout(x)

        x = self.layers[-1](x, edge_index, edge_weight)
        return x
