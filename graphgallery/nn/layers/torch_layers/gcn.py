import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.nn import Module

class GraphConvolution(Module):
    def __init__(self, input_channels, output_channels, use_bias=False):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel = Parameter(torch.FloatTensor(input_channels, output_channels))
        
        if use_bias:
            self.bias = Parameter(torch.FloatTensor(output_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.kernel.size(1))
        self.kernel.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

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
               + str(self.input_channels) + ' -> ' \
               + str(self.output_channels) + ')'
    
    
class ListModule(Module):
    """
    Abstract list layer class.
    """
    def __init__(self, *args):
        """
        Module initializing.
        """
        super().__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)    