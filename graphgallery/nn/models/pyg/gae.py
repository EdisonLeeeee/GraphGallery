import torch
import torch.nn as nn
from graphgallery.nn.layers.pytorch import Sequential, activations, InnerProductDecoder
from torch_geometric.nn import GCNConv


class GAE(nn.Module):
    def __init__(self,
                 in_features, *,
                 out_features=16,
                 hids=[32],
                 acts=['relu'],
                 dropout=0.5,
                 bias=False):
        super().__init__()

        encoder = []
        for hid, act in zip(hids, acts):
            encoder.append(GCNConv(in_features,
                                   hid,
                                   cached=True,
                                   bias=bias,
                                   normalize=True))
            encoder.append(activations.get(act))
            encoder.append(nn.Dropout(dropout))
            in_features = hid

        encoder.append(GCNConv(in_features,
                               out_features,
                               cached=True,
                               bias=bias,
                               normalize=True))

        self.encoder = Sequential(*encoder)
        self.decoder = InnerProductDecoder()

    def forward(self, x, edge_index, edge_weight=None):
        z = self.encoder(x, edge_index, edge_weight)
        return z

    def cache_clear(self):
        for conv in self.encoder:
            if isinstance(conv, GCNConv):
                conv._cached_edge_index = None
                conv._cached_adj_t = None


class VGAE(nn.Module):
    def __init__(self,
                 in_features, *,
                 out_features=16,
                 hids=[32],
                 acts=['relu'],
                 dropout=0.5,
                 bias=False):
        super().__init__()

        conv = []
        for hid, act in zip(hids, acts):
            conv.append(GCNConv(in_features,
                                hid,
                                cached=True,
                                bias=bias,
                                normalize=True))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_features = hid

        self.mu_conv = GCNConv(in_features, out_features, bias=bias)
        self.logstd_conv = GCNConv(in_features, out_features, bias=bias)
        self.conv = Sequential(*conv)
        self.decoder = InnerProductDecoder()

    def forward(self, x, edge_index, edge_weight=None):
        h = self.conv(x, edge_index, edge_weight)
        mu = self.mu_conv(h, edge_index, edge_weight)
        if self.training:
            logstd = self.logstd_conv(h, edge_index, edge_weight)
            std = torch.exp(logstd)
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mu)
            return z, mu, logstd
        else:
            return mu

    def cache_clear(self):
        for conv in self.encoder:
            if isinstance(conv, GCNConv):
                conv._cached_edge_index = None
                conv._cached_adj_t = None
