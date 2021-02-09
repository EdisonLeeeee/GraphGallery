import torch
import torch.nn as nn
from graphgallery.nn.init.pytorch import uniform, zeros
from ..get_activation import get_activation


class PropConvolution(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=1,
                 K=10,
                 bias=False,
                 activation=None):
        super().__init__()
        assert out_channels == 1, "'out_channels' must be 1"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = get_activation(activation)
        self.w = nn.Linear(in_channels, out_channels, bias=bias)
        self.K = K

    def reset_parameters(self):
        self.w.reset_parameters()

    def forward(self, x, adj):

        propagations = [x]
        for _ in range(self.K):
            x = adj.mm(x)
            propagations.append(x)

        h = torch.stack(propagations, dim=1)
        retain_score = self.w(h)
        retain_score = self.activation(retain_score).permute(0, 2, 1).contiguous()
        out = (retain_score @ h).squeeze(1)
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, K={self.K})"
