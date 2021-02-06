import torch
import torch.nn as nn
from ..get_activation import get_activation


class TAGConvolution(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 K=3,
                 use_bias=True,
                 activation=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.activation = get_activation(activation)
        self.w = nn.Linear(in_channels * (self.K + 1),
                           out_channels, bias=use_bias)

    def reset_parameters(self):
        self.w.reset_parameters()

    def forward(self, x, adj):

        out = x
        xs = [x]
        for _ in range(self.K):
            out = adj.mm(out)
            xs.append(out)
        out = self.w(torch.cat(xs, dim=-1))
        return self.activation(out)

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.K)
