import math
import torch
import torch.nn as nn

from graphgallery.nn.init.pytorch import uniform, zeros
from ..get_activation import get_activation


class MedianConvolution(nn.Module):
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

    def forward(self, x, nbrs):
        h = self.w(x)
        aggregation = []
        for nbr in nbrs:
            message, _ = torch.median(h[nbr], dim=0)
            aggregation.append(message)

        output = torch.stack(aggregation)

        return self.activation(output)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels})"
