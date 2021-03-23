import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.layers.pytorch import MeanAggregator, Sequential, activations
from graphgallery.nn.metrics.pytorch import Accuracy

_AGG = {'mean': MeanAggregator,
        # 'gcn': GCNAggregator,
        # 'median': MedianAggregator,
        # 'mediangcn': MedianGCNAggregator
        }

class GraphSAGE(TorchKeras):
    def __init__(self, in_features, out_features,
                 hids=[32], acts=['relu'], dropout=0.5,
                 weight_decay=5e-4, lr=0.01, bias=False,
                 aggregator='mean', output_normalize=False, 
                 num_samples=[15, 5], concat=True):
        
        super().__init__()
        Agg = _AGG.get(aggregator, None)
        if not Agg:
            raise ValueError(
                f"Invalid value of 'aggregator', allowed values {tuple(_AGG.keys())}, but got '{aggregator}'.")

        self.output_normalize = output_normalize
        self.num_samples = num_samples

        aggregators, act_layers = nn.ModuleList(), nn.ModuleList()
        for hid, act in zip(hids, acts):
            aggregators.append(Agg(in_features, hid, concat=concat, bias=bias))
            act_layers.append(activations.get(act))
            in_features = hid * 2 if concat else hid
            
        aggregators.append(Agg(in_features, out_features, bias=bias))

        self.aggregators = aggregators
        self.dropout = nn.Dropout(dropout)
        self.acts = act_layers

        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay),
                     metrics=[Accuracy()])

    def forward(self, x, nodes, *neighbors):
        num_samples = self.num_samples

        h = [x[node] for node in [nodes, *neighbors]]
        for agg_i, aggregator in enumerate(self.aggregators):
            attribute_shape = h[0].shape[-1]
            for hop in range(len(num_samples) - agg_i):
                neighbor_shape = [-1, num_samples[hop], attribute_shape]
                # x, neigh_x
                h[hop] = aggregator(h[hop], h[hop + 1].view(neighbor_shape))
                if hop != len(num_samples) - 1:
                    h[hop] = self.dropout(self.acts[hop](h[hop]))
            h.pop()
        h = h[0]
        if self.output_normalize:
            h = F.normalize(h, dim=1,p=2)

        return h
