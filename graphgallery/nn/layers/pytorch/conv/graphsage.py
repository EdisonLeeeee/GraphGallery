import torch
import torch.nn as nn
from graphgallery.nn.init.pytorch import glorot_uniform, zeros

class MeanAggregator(nn.Module):
    def __init__(self, in_features, out_features, 
                 agg_method='mean',
                 concat=False, use_bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.output_dim = out_features * 2 if concat else out_features
        self.use_bias = use_bias
        self.concat = concat

        # TODO not sure
        # weight_decay放哪里？
        self.agg_method = agg_method
        self.aggregator = {'mean': torch.mean, 'sum': torch.sum,
                           'max': torch.max, 'min': torch.min}[agg_method]

        self.kernel_self = nn.Parameter(torch.FloatTensor(in_features, out_features), requires_grad=True)
        self.kernel_neigh = nn.Parameter(torch.FloatTensor(in_features, out_features), requires_grad=True)
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        glorot_uniform(self.kernel_self)
        glorot_uniform(self.kernel_neigh)
        if self.use_bias:
            zeros(self.bias)

    def forward(self, x, neigh_x):
        neigh_x = self.aggregator(neigh_x, axis=1)
        
        x = x.mm(self.kernel_self)
        neigh_x = neigh_x.mm(self.kernel_neigh)
        output = torch.cat([x, neigh_x], 1) if self.concat else x + neigh_x

        if self.use_bias:
            output += self.bias

        return output

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"
