import torch.nn as nn
import torch.nn.functional as F

from torch import optim

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.layers.pytorch import SAGEAggregator, activations
from graphgallery.nn.metrics.pytorch import Accuracy

_AGG = {'mean': SAGEAggregator, }


class GraphSAGE(TorchKeras):
    def __init__(self, in_features, out_features,
                 hids=[32], acts=['relu'], dropout=0.5,
                 weight_decay=5e-4, lr=0.01, bias=False,
                 aggregator='mean', output_normalize=False,
                 sizes=[15, 5], concat=True):

        super().__init__()
        Agg = _AGG.get(aggregator, None)
        if not Agg:
            raise ValueError(
                f"Invalid value of 'aggregator', allowed values {tuple(_AGG.keys())}, but got '{aggregator}'.")

        self.output_normalize = output_normalize
        self.sizes = sizes
        assert len(sizes) == len(hids) + 1

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
        # print(nodes, neighbors)
        # print(neighbors)
        # print(x.size())
        sizes = self.sizes

        h = [x[node] for node in [nodes, *neighbors]]
        for i, aggregator in enumerate(self.aggregators):
            dim = h[0].size(-1)
            for j in range(len(sizes) - i):
                # x, neigh_x
                h[j] = aggregator(h[j], h[j + 1].view(-1, sizes[j], dim))
                if i != len(sizes) - 1:
                    h[j] = self.dropout(self.acts[i](h[j]))
            h.pop()
        h = h[0]
        if self.output_normalize:
            h = F.normalize(h, dim=1, p=2)
        return h
