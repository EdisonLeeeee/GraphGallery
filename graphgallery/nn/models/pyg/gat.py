import torch
import torch.nn.functional as F

from torch.nn import Module, ModuleList, Dropout
from torch import optim
from torch_geometric.nn import GATConv

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.layers.pytorch.get_activation import get_activation


class GAT(TorchKeras):

    def __init__(self, in_channels,
                 out_channels, hiddens=[8],
                 n_heads=[8], activations=['elu'],
                 dropout=0.6, weight_decay=5e-4,
                 lr=0.01, use_bias=True):

        super().__init__()

        layers = ModuleList()
        acts = []
        paras = []

        inc = in_channels
        pre_head = 1
        for hidden, n_head, activation in zip(hiddens, n_heads, activations):
            layer = GATConv(inc * pre_head, hidden, heads=n_head,
                            bias=use_bias, dropout=dropout)
            layers.append(layer)
            acts.append(get_activation(activation))
            paras.append(dict(params=layer.parameters(), weight_decay=weight_decay))
            inc = hidden
            pre_head = n_head

        layer = GATConv(inc * pre_head, out_channels, heads=1, 
                        bias=use_bias, concat=False,
                        dropout=dropout)
        layers.append(layer)
        # do not use weight_decay in the final layer
        paras.append(dict(params=layer.parameters(), weight_decay=0.))

        self.acts = acts
        self.layers = layers        
        self.optimizer = optim.Adam(paras, lr=lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.dropout = Dropout(dropout)

    def forward(self, inputs):
        x, edge_index, edge_weight, idx = inputs
        x = self.dropout(x)
        
        for layer, act in zip(self.layers, self.acts):
            x = act(layer(x, edge_index))
            x = self.dropout(x)

        x = self.layers[-1](x, edge_index)
        return x[idx]
