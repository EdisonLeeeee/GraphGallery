import torch
import math
import numpy as np
from torch.nn import Module, Parameter


class SGConvolution(Module):
    def __init__(self, order=1, **kwargs):
        """
        Initialize the order.

        Args:
            self: (todo): write your description
            order: (int): write your description
        """
        super().__init__()
        self.order = order

    def forward(self, inputs):
        """
        Forward computation of the graph.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        x, adj = inputs

        for _ in range(self.order):
            x = torch.spmm(adj, x)

        return x
    
    def reset_parameters(self):
        """
        Reset the parameters.

        Args:
            self: (todo): write your description
        """
        ...
        
    def __repr__(self):
        """
        Return a repr representation of a repr__.

        Args:
            self: (todo): write your description
        """
        return f"{self.__class__.__name__} (order={self.order})"
