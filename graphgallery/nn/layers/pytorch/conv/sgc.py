import torch
from torch.nn import Module, Parameter


class SGConvolution(Module):
    def __init__(self, order=1, **kwargs):
        super().__init__()
        self.order = order

    def forward(self, inputs):
        x, adj = inputs

        for _ in range(self.order):
            x = torch.spmm(adj, x)

        return x
    
    def reset_parameters(self):
        ...
        
    def __repr__(self):
        return f"{self.__class__.__name__} (order={self.order})"
