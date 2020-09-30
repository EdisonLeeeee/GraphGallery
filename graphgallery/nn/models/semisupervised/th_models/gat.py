import torch
import torch.nn.functional as F

from torch.nn import Module, ModuleList
from torch import optim

from graphgallery.nn.models import TorchKerasModel
from graphgallery.nn.models.get_activation import get_activation
from graphgallery.nn.layers.th_layers import GraphAttention, SparseGraphAttention


class GAT(TorchKerasModel):

    def __init__(self, in_channels, hiddens,
                 out_channels, n_heads=[8], activations=['elu'],
                 dropouts=[0.6], l2_norms=[5e-4],
                 lr=0.01, use_bias=True):

        super().__init__()

        # save for later usage
        self.dropouts = dropouts

        self.layers = ModuleList()
        self.acts = []
        paras = []

        inc = in_channels
        pre_head = 1
        for hidden, n_head, act, l2_norm in zip(hiddens, n_heads, activations, l2_norms):
            layer = SparseGraphAttention(inc * pre_head, hidden, attn_heads=n_head, reduction='concat', use_bias=use_bias)
            self.layers.append(layer)
            self.acts.append(get_activation(act))
            paras.append(dict(params=layer.parameters(), weight_decay=l2_norm))
            inc = hidden
            pre_head = n_head

        layer = SparseGraphAttention(inc * pre_head, out_channels, attn_heads=1, reduction='average', use_bias=use_bias)
        self.layers.append(layer)
        # do not use weight_decay in the final layer
        paras.append(dict(params=layer.parameters(), weight_decay=0.))

        self.optimizer = optim.Adam(paras, lr=lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, inputs):
        x, adj, idx = inputs

        for i in range(len(self.layers) - 1):
            x = F.dropout(x, self.dropouts[i], training=self.training)
            act = self.acts[i]
            x = act(self.layers[i]([x, adj]))
            
        # add extra dropout
        x = F.dropout(x, self.dropouts[-1], training=self.training)
        x = self.layers[-1]([x, adj])  # last layer

        return x[idx]