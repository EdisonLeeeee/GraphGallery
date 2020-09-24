import torch
import math
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn import Module
from graphgallery.nn.init import kaiming_uniform, zeros

# TODO: change dtypes of trainable weights based on `floax`
class GraphConvolution(Module):
    def __init__(self, in_channels, out_channels, use_bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = Parameter(torch.Tensor(in_channels, out_channels))

        if use_bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_uniform(self.kernel, a=math.sqrt(5))
        zeros(self.bias)

    def forward(self, inputs):
        x, adj = inputs
        h = torch.spmm(x, self.kernel)
        output = torch.spmm(adj, h)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_channels) + ' -> ' \
            + str(self.out_channels) + ')'
