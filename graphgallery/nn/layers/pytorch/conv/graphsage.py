import torch
import torch.nn as nn


class SAGEAggregator(nn.Module):
    def __init__(self, in_features, out_features,
                 agg_method='mean',
                 concat=False,
                 bias=False):

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.agg_method = agg_method
        self.aggregator = {'mean': torch.mean, 'sum': torch.sum,
                           'max': torch.max, 'min': torch.min}[agg_method]

        self.lin_l = nn.Linear(in_features, out_features, bias=bias)
        self.lin_r = nn.Linear(in_features, out_features, bias=bias)

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, neigh_x):
        if not isinstance(x, torch.Tensor):
            x = torch.cat(x, dim=0)
        if not isinstance(neigh_x, torch.Tensor):
            neigh_x = torch.cat([self.aggregator(h, dim=1) for h in neigh_x], dim=0)
        else:
            neigh_x = self.aggregator(neigh_x, dim=1)

        neigh_x = self.lin_r(neigh_x)
        x = self.lin_l(x)
        out = torch.cat([x, neigh_x], dim=1) if self.concat else x + neigh_x
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"
