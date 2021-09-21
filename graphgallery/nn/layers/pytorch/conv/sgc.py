import torch
from torch.nn import Module


class SGConv(Module):
    def __init__(self, K=2, **kwargs):
        super().__init__()
        self.K = K

    def forward(self, x, adj):

        for _ in range(self.K):
            x = torch.spmm(adj, x)

        return x

    def reset_parameters(self):
        pass

    def extra_repr(self):
        return f"K={self.K}"
