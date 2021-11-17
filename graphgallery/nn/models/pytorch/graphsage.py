import torch.nn as nn
import torch.nn.functional as F

from graphgallery.nn.layers.pytorch import SAGEAggregator, activations


class GraphSAGE(nn.Module):
    def __init__(self, in_features, out_features,
                 hids=[32], acts=['relu'], dropout=0.5, bias=False,
                 aggregator='mean', output_normalize=False,
                 sizes=[15, 5], concat=True):

        super().__init__()
        self.sizes = sizes
        assert len(sizes) == len(hids) + 1

        aggregators, act_layers = nn.ModuleList(), nn.ModuleList()
        for hid, act in zip(hids, acts):
            aggregators.append(SAGEAggregator(in_features, hid, concat=concat, bias=bias, agg_method=aggregator))
            act_layers.append(activations.get(act))
            in_features = hid * 2 if concat else hid

        aggregators.append(SAGEAggregator(in_features, out_features, bias=bias, agg_method=aggregator))

        self.aggregators = aggregators
        self.dropout = nn.Dropout(dropout)
        self.acts = act_layers
        self.output_normalize = output_normalize

    def forward(self, x, nodes, *neighbors):
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
