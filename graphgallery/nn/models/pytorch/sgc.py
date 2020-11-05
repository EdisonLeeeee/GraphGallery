import torch
import torch.nn.functional as F

from torch.nn import Module, ModuleList, Dropout, Linear
from torch import optim

from graphgallery.nn.models import TorchKeras


class SGC(TorchKeras):

    def __init__(self, in_channels,
                 out_channels, 
                 hiddens=[], 
                 activations=[], 
                 dropout=0.5,
                 l2_norm=5e-5,
                 lr=0.2, use_bias=False):
        super().__init__()
        
        if len(hiddens) != len(activations):
            raise RuntimeError(f"Arguments 'hiddens' and 'activations' should have the same length."
                               " Or you can set both of them to `[]`.")

        self.layers = ModuleList()
        
        paras = []
        inc = in_channels
        for hidden, activation in zip(hiddens, activations):
            layer = Linear(inc, hidden, bias=use_bias)
            paras.append(dict(params=layer.parameters(), weight_decay=l2_norm))
            self.layers.append(layer)
            inc = hidden
            
        layer = Linear(inc, out_channels, bias=use_bias)
        self.layers.append(layer)
        # do not use weight_decay in the final layer
        paras.append(dict(params=layer.parameters(), weight_decay=l2_norm))
        
        self.dropout = Dropout(dropout)
        self.optimizer = optim.Adam(paras, lr=lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, inputs):
        x = inputs

        for layer in self.layers:
            x = self.dropout(x)
            x = layer(x)
            
        return x
