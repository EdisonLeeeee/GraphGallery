import torch
from torch import optim
from torch.nn import Module, ModuleList, Dropout

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.layers.pytorch.get_activation import get_activation
from graphgallery.nn.metrics.pytorch import Accuracy

from dgl.nn.pytorch import GATConv


class GAT(TorchKeras):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hiddens=[8],
                 n_heads=[8],
                 activations=['elu'],
                 dropout=0.6,
                 weight_decay=5e-4,
                 lr=0.01):

        super().__init__()

        layers = ModuleList()
        paras = []

        inc = in_channels
        pre_head = 1
        for hidden, n_head, activation in zip(hiddens, n_heads, activations):
            layer = GATConv(inc * pre_head,
                            hidden,
                            activation=get_activation(activation),
                            num_heads=n_head,
                            feat_drop=dropout,
                            attn_drop=dropout)
            layers.append(layer)
            paras.append(
                dict(params=layer.parameters(), weight_decay=weight_decay))
            inc = hidden
            pre_head = n_head

        layer = GATConv(inc * pre_head,
                        out_channels,
                        num_heads=1,
                        feat_drop=dropout,
                        attn_drop=dropout)
        layers.append(layer)
        # do not use weight_decay in the final layer
        paras.append(dict(params=layer.parameters(), weight_decay=0.))

        self.layers = layers
        self.dropout = Dropout(dropout)
        self.compile(loss=torch.nn.CrossEntropyLoss(),
                     optimizer=optim.Adam(paras, lr=lr),
                     metrics=[Accuracy()])

    def forward(self, inputs):
        x, g, idx = inputs
        for layer in self.layers[:-1]:
            x = layer(g, x).flatten(1)
            x = self.dropout(x)

        x = self.layers[-1](g, x).mean(1)
        return x[idx]
