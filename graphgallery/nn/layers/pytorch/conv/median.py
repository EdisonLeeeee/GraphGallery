import math
import torch
import torch.nn as nn


class MedianConvolution(nn.Module):
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

    def forward(self, x, nbrs):
        h = self.w(x)
        aggregation = []
        for nbr in nbrs:
            message, _ = torch.median(h[nbr], dim=0)
            aggregation.append(message)

        output = torch.stack(aggregation)

        return output

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels})"
