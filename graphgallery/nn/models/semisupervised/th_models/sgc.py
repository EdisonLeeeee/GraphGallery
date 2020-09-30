import torch
import torch.nn.functional as F

from torch.nn import Module, ModuleList, Dropout, Linear
from torch import optim

from graphgallery.nn.models import TorchKerasModel


class SGC(TorchKerasModel):

    def __init__(self, in_channels,
                 out_channels, hiddens=[], activations=[],
                 l2_norms=[5e-5], dropout=0.5,
                 lr=0.2, use_bias=False):
        super().__init__()
        
        if len(hiddens) != len(activations):
            raise RuntimeError(f"Arguments 'hiddens' and 'activations' should have the same length."
                               " Or you can set both of them to `[]`.")

        self.layers = ModuleList()
        
        paras = []
        inc = in_channels
        for hidden, activation, l2_norm in zip(hiddens, activations, l2_norms):
            layer = Linear(inc, hidden, bias=use_bias)
            paras.append(dict(params=layer.parameters(), weight_decay=l2_norm))
            self.layers.append(layer)
            inc = hidden
            
        layer = Linear(inc, out_channels, bias=use_bias)
        self.layers.append(layer)
        # do not use weight_decay in the final layer
        paras.append(dict(params=layer.parameters(), weight_decay=l2_norms[-1]))
        
        self.dropout = Dropout(dropout)
        self.optimizer = optim.Adam(paras, lr=lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, inputs):
        x = inputs

        for layer in self.layers:
            x = self.dropout(x)
            x = layer(x)
            
        return x
