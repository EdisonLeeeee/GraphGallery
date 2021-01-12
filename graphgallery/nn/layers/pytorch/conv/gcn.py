import torch
import torch.nn as nn
from graphgallery.nn.init.pytorch import uniform, zeros
from ..get_activation import get_activation


# TODO: change dtypes of trainable weights based on `floax`
class GraphConvolution(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_bias=False,
                 activation=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = get_activation(activation)
        self.w = nn.Linear(in_channels, out_channels, bias=use_bias)

    def reset_parameters(self):
        self.w.reset_parameters()

    def forward(self, x, adj=None):
        out = self.w(x)
        if adj is not None:
            out = adj.mm(out)

        return self.activation(out)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels} -> {self.out_channels})"
