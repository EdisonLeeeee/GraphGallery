import torch
import torch.nn as nn


class TAGConv(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 K=3,
                 bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = K
        self.w = nn.Linear(in_features * (self.K + 1),
                           out_features, bias=bias)

    def reset_parameters(self):
        self.w.reset_parameters()

    def forward(self, x, adj):

        out = x
        xs = [x]
        for _ in range(self.K):
            out = adj.mm(out)
            xs.append(out)
        out = self.w(torch.cat(xs, dim=-1))
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features}, K={self.K})"
