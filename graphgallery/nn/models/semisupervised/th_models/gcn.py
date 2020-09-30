import torch
import torch.nn.functional as F

from torch.nn import Module, ModuleList
from torch import optim

from graphgallery.nn.models import TorchKerasModel
from graphgallery.nn.models.get_activation import get_activation
from graphgallery.nn.layers.th_layers import GraphConvolution


class GCN(TorchKerasModel):

    def __init__(self, in_channels, hiddens,
                 out_channels, activations=['relu'],
                 dropouts=[0.5], l2_norms=[5e-4],
                 lr=0.01, use_bias=False):

        super().__init__()

        # save for later usage
        self.dropouts = dropouts

        self.layers = ModuleList()
        self.acts = []
        paras = []

        # use ModuleList to create layers with different size
        inc = in_channels
        for hidden, act, l2_norm in zip(hiddens, activations, l2_norms):
            layer = GraphConvolution(inc, hidden, use_bias=use_bias)
            self.layers.append(layer)
            self.acts.append(get_activation(act))
            paras.append(dict(params=layer.parameters(), weight_decay=l2_norm))
            inc = hidden

        layer = GraphConvolution(inc, out_channels, use_bias=use_bias)
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
