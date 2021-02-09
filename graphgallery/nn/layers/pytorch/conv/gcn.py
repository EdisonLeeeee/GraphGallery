import torch.nn as nn


class GraphConvolution(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w = nn.Linear(in_channels, out_channels, bias=bias)

    def reset_parameters(self):
        self.w.reset_parameters()

    def forward(self, x, adj=None):
        out = self.w(x)
        if adj is not None:
            out = adj.mm(out)
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels})"
