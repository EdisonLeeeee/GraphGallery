import torch
import torch.nn.functional as F

from torch.nn import Module, ModuleList, Dropout
from torch import optim

from graphgallery.nn.models import TorchKerasModel
from graphgallery.nn.layers.pytorch import GraphAttention, SparseGraphAttention


class GAT(TorchKerasModel):

    def __init__(self, in_channels,
                 out_channels, hiddens=[8],
                 n_heads=[8], activations=['elu'],
                 dropout=0.6, l2_norm=5e-4,
                 lr=0.01, use_bias=True):

        super().__init__()

        self.layers = ModuleList()
        paras = []

        inc = in_channels
        pre_head = 1
        for hidden, n_head, activation in zip(hiddens, n_heads, activations):
            layer = SparseGraphAttention(inc * pre_head, hidden, activation=activation,
                                         attn_heads=n_head, reduction='concat', use_bias=use_bias)
            self.layers.append(layer)
            paras.append(dict(params=layer.parameters(), weight_decay=l2_norm))
            inc = hidden
            pre_head = n_head

        layer = SparseGraphAttention(inc * pre_head, out_channels, attn_heads=1, reduction='average', use_bias=use_bias)
        self.layers.append(layer)
        # do not use weight_decay in the final layer
        paras.append(dict(params=layer.parameters(), weight_decay=0.))

        self.optimizer = optim.Adam(paras, lr=lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.dropout = Dropout(dropout)

    def forward(self, inputs):
        x, adj, idx = inputs

        for layer in self.layers:
            x = self.dropout(x)
            x = layer([x, adj])

        return x[idx]
