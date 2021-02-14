import torch
import torch.nn as nn


class PropConv(nn.Module):
    def __init__(self,
                 in_features,
                 out_features=1,
                 K=10,
                 bias=False):
        super().__init__()
        assert out_features == 1, "'out_features' must be 1"
        self.in_features = in_features
        self.out_features = out_features
        self.w = nn.Linear(in_features, out_features, bias=bias)
        self.K = K

    def reset_parameters(self):
        self.w.reset_parameters()

    def forward(self, x, adj):

        propagations = [x]
        for _ in range(self.K):
            x = adj.mm(x)
            propagations.append(x)

        h = torch.stack(propagations, dim=1)
        retain_score = self.w(h).permute(0, 2, 1).contiguous()
        out = (retain_score @ h).squeeze(1)
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features}, K={self.K})"
