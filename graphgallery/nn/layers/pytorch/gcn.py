import torch
import math
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn import Module
from graphgallery.nn.init import uniform, zeros
from .get_activation import get_activation

# TODO: change dtypes of trainable weights based on `floax`
class GraphConvolution(Module):
    def __init__(self, in_channels, out_channels, use_bias=False, activation=None):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            in_channels: (int): write your description
            out_channels: (int): write your description
            use_bias: (bool): write your description
            activation: (str): write your description
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = get_activation(activation)
        self.kernel = Parameter(torch.Tensor(in_channels, out_channels))

        if use_bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the kernel.

        Args:
            self: (todo): write your description
        """
        uniform(self.kernel)
        zeros(self.bias)

    def forward(self, inputs):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        x, adj = inputs
        h = torch.mm(x, self.kernel)
        output = torch.spmm(adj, h)

        if self.bias is not None:
            output += self.bias
            
        return self.activation(output)

    def __repr__(self):
        """
        Return a representation of this channel.

        Args:
            self: (todo): write your description
        """
        return self.__class__.__name__ + ' (' \
            + str(self.in_channels) + ' -> ' \
            + str(self.out_channels) + ')'
