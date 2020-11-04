import torch
import torch.nn.functional as F
from torch import optim
from torch.nn import Module, ModuleList, Dropout

from graphgallery.nn.models import TorchKerasModel
from graphgallery.nn.layers.pytorch.get_activation import get_activation

from torch_geometric.nn import GCNConv

class GCN(TorchKerasModel):
    def __init__(self, in_channels, out_channels,
                 hiddens=[16],
                 activations=['relu'],
                 dropout=0.5,
                 l2_norm=5e-4,
                 lr=0.01, use_bias=True):

        super().__init__()

        paras = []
        acts = []
        layers = ModuleList()

        # use ModuleList to create layers with different size
        inc = in_channels
        for hidden, activation in zip(hiddens, activations):
            layer = GCNConv(inc, hidden, cached=True, bias=use_bias)
            layers.append(layer)
            paras.append(dict(params=layer.parameters(), weight_decay=l2_norm))
            acts.append(get_activation(activation))
            inc = hidden

        layer = GCNConv(inc, out_channels, cached=True, bias=use_bias)
        layers.append(layer)
        # do not use weight_decay in the final layer
        paras.append(dict(params=layer.parameters(), weight_decay=0.))

        self.acts = acts
        self.layers = layers
        self.dropout = Dropout(dropout)
        self.optimizer = optim.Adam(paras, lr=lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
    def forward(self, inputs):
        x, edge_index, edge_weight, idx = inputs
        
        for layer, act in zip(self.layers, self.acts):
            x = act(layer(x, edge_index, edge_weight))
            x = self.dropout(x)
            
        x = self.layers[-1](x, edge_index, edge_weight)
        return x[idx]