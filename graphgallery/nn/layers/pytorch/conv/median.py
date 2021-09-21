import math
import torch
import torch.nn as nn


class MedianConv(nn.Module):
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

    def forward(self, x, nbrs):
        h = self.w(x)
        aggregation = []
        for nbr in nbrs:
            message, _ = torch.median(h[nbr], dim=0)
            aggregation.append(message)

        output = torch.stack(aggregation)

        return output

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"
