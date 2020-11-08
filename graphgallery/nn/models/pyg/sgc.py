import torch
import torch.nn.functional as F
from torch import optim
from torch.nn import Module, ModuleList, Dropout, Linear

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.layers.pytorch.get_activation import get_activation

from torch_geometric.nn import SGConv


class SGC(TorchKeras):

    def __init__(self, in_channels,
                 out_channels,
                 hiddens=[],
                 activations=[],
                 K=2,
                 dropout=0.5,
                 weight_decay=5e-5,
                 lr=0.2, use_bias=False):
        super().__init__()

        if hiddens or activations:
            raise RuntimeError(f"Arguments 'hiddens' and 'activations' are not supported to use in SGC (PyG backend).")

        conv = SGConv(in_channels, out_channels, bias=use_bias, K=K, cached=True, add_self_loops=False)
        self.conv = conv
        self.dropout = Dropout(dropout)
        self.optimizer = optim.Adam(conv.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, inputs):
        x, edge_index, edge_weight, idx = inputs
        x = self.dropout(x)
        x = self.conv(x, edge_index, edge_weight)
        return x[idx]
