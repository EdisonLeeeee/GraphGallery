import torch.nn as nn
from torch.nn import Module, Parameter


class TrainableSGConv(Module):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=False,
                 K=2,
                 cached=True,
                 **kwargs):

        super().__init__()
        self.K = K
        self.w = nn.Linear(in_features, out_features, bias=bias)
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
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features}, K={self.K})"
