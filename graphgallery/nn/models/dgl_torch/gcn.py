import torch
import torch.nn.functional as F
from torch import optim
from torch.nn import Module, ModuleList, Dropout

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.layers.pytorch.get_activation import get_activation

from dgl.nn.pytorch import GraphConv

class GCN(TorchKeras):
    def __init__(self, in_channels, out_channels,
                 hiddens=[16],
                 activations=['relu'],
                 dropout=0.5,
                 l2_norm=5e-4,
                 lr=0.01, use_bias=True):
        
        super().__init__()

        self.layers = ModuleList()

        inc = in_channels
        for hidden, activation in zip(hiddens, activations):
            act = get_activation(activation)
            layer = GraphConv(inc, hidden, activation=act, bias=use_bias)
            self.layers.append(layer)
            inc = hidden
        # output layer
        self.layers.append(GraphConv(inc, out_channels))

        self.dropout = Dropout(p=dropout)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr,
                                          weight_decay=l2_norm)

    def forward(self, inputs):
        x, g, indx = inputs

        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
            x = layer(g, x)

        return x[indx]

