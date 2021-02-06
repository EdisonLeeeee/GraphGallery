import torch
from torch.nn import Module, Parameter


class SGConvolution(Module):
    def __init__(self, order=2, **kwargs):
        super().__init__()
        self.order = order

    def forward(self, x, adj):

        for _ in range(self.order):
            x = adj.mm(x)

        return x

    def reset_parameters(self):
        pass

    def extra_repr(self):
        return f"order={self.order}"
