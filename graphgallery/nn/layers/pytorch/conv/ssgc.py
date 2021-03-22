import torch
from torch.nn import Module


class SSGConv(Module):
    def __init__(self, K=16, alpha=0.1, **kwargs):
        super().__init__()
        assert K>0
        self.K = K
        self.alpha = alpha

    def forward(self, x, adj):
        x_in = x
        x_out = torch.zeros_like(x)
        for _ in range(self.K):
            x =  torch.spmm(adj, x)
            x_out += (1 - self.alpha) * x
        x_out /= self.K
        x_out += self.alpha * x_in
        return x_out

    def reset_parameters(self):
        pass

    def extra_repr(self):
        return f"K={self.K}, alpha={self.alpha}"
