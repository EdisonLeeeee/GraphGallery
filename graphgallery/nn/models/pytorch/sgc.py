import torch
import torch.nn.functional as F

from torch.nn import Module, ModuleList, Dropout, Linear
from torch import optim

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.layers.pytorch.get_activation import get_activation


class SGC(TorchKeras):

    def __init__(self, in_channels,
                 out_channels,
                 hiddens=[],
                 activations=[],
                 dropout=0.5,
                 weight_decay=5e-5,
                 lr=0.2, use_bias=False):
        super().__init__()

        if len(hiddens) != len(activations):
            raise RuntimeError(f"Arguments 'hiddens' and 'activations' should have the same length."
                               " Or you can set both of them to `[]`.")

        layers = ModuleList()
        acts = []
        paras = []
        inc = in_channels
        for hidden, activation in zip(hiddens, activations):
            layer = Linear(inc, hidden, bias=use_bias)
            paras.append(dict(params=layer.parameters(), weight_decay=weight_decay))
            layers.append(layer)
            inc = hidden
            acts.append(get_activation(activation))

        layer = Linear(inc, out_channels, bias=use_bias)
        layers.append(layer)
        paras.append(dict(params=layer.parameters(), weight_decay=weight_decay))

        self.layers = layers
        self.acts = acts
        self.dropout = Dropout(dropout)
        self.optimizer = optim.Adam(paras, lr=lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, inputs):
        x = inputs

        for layer, act in zip(self.layers, self.acts):
            x = self.dropout(x)
            x = act(layer(x))

        x = self.layers[-1](x)
        return x
