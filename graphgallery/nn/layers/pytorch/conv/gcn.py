import torch.nn as nn


class GCNConv(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w = nn.Linear(in_features, out_features, bias=bias)

    def reset_parameters(self):
        self.w.reset_parameters()

    def forward(self, x, adj=None):
        out = self.w(x)
        if adj is not None:
            out = adj.mm(out)
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"
