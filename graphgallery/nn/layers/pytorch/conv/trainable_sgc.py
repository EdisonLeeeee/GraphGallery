import torch.nn as nn
from torch.nn import Module, Parameter


class TrainableSGConvolution(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_bias=False,
                 K=2,
                 cached=True,
                 **kwargs):

        super().__init__()
        self.K = K
        self.w = nn.Linear(in_channels, out_channels, bias=use_bias)
        self.cache = None
        self.cached = cached

    def forward(self, x, adj):

        if self.cache is None or not self.cached:
            for _ in range(self.K):
                x = adj.mm(x)
            self.cache = x
        else:
            x = self.cache

        return self.w(x)

    def reset_parameters(self):
        self.w.reset_parameters()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, K={self.K})"
