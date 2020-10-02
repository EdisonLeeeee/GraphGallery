import torch
import torch.nn.functional as F

from torch.nn import Module, ModuleList, Dropout
from torch import optim

from graphgallery.nn.models import TorchKerasModel
from graphgallery.nn.layers.th_layers import GraphConvolution


class GCN(TorchKerasModel):

    def __init__(self, in_channels, out_channels, 
                 hiddens=[16],
                 activations=['relu'],
                 dropout=0.5,
                 l2_norm=5e-4, 
                 lr=0.01, use_bias=False):

        super().__init__()

        self.layers = ModuleList()
        paras = []

        # use ModuleList to create layers with different size
        inc = in_channels
        for hidden, activation in zip(hiddens, activations):
            layer = GraphConvolution(inc, hidden, activation=activation, use_bias=use_bias)
            self.layers.append(layer)
            paras.append(dict(params=layer.parameters(), weight_decay=l2_norm))
            inc = hidden

        layer = GraphConvolution(inc, out_channels, use_bias=use_bias)
        self.layers.append(layer)
        # do not use weight_decay in the final layer
        paras.append(dict(params=layer.parameters(), weight_decay=0.))
        
        self.dropout = Dropout(dropout)
        self.optimizer = optim.Adam(paras, lr=lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, inputs):
        x, adj, idx = inputs

        for layer in self.layers:
            x = self.dropout(x)
            x = layer([x, adj])
            
        return x[idx]
