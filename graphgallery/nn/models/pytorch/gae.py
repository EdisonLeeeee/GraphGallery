import torch
import torch.nn as nn
from graphgallery.nn.layers.pytorch import GCNConv, Sequential, activations, InnerProductDecoder


class GAE(nn.Module):
    def __init__(self,
                 in_features,
                 *,
                 out_features=16,
                 hids=[32],
                 acts=['relu'],
                 dropout=0.,
                 bias=False):
        super().__init__()
        encoder = []
        encoder.append(nn.Dropout(dropout))
        for hid, act in zip(hids, acts):
            encoder.append(GCNConv(in_features,
                                   hid,
                                   bias=bias))
            encoder.append(activations.get(act))
            encoder.append(nn.Dropout(dropout))
            in_features = hid
        encoder.append(GCNConv(in_features, out_features, bias=bias))
        encoder = Sequential(*encoder)

        self.encoder = encoder
        self.decoder = InnerProductDecoder()

    def forward(self, x, adj):
        z = self.encoder(x, adj)
        return z


class VGAE(nn.Module):
    def __init__(self,
                 in_features,
                 *,
                 out_features=16,
                 hids=[32],
                 acts=['relu'],
                 dropout=0.,
                 bias=False):
        super().__init__()
        conv = []
        conv.append(nn.Dropout(dropout))
        for hid, act in zip(hids, acts):
            conv.append(GCNConv(in_features,
                                hid,
                                bias=bias))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_features = hid

        self.mu_conv = GCNConv(in_features, out_features, bias=bias)
        self.logstd_conv = GCNConv(in_features, out_features, bias=bias)
        self.conv = Sequential(*conv)
        self.decoder = InnerProductDecoder()

    def forward(self, x, adj):
        h = self.conv(x, adj)
        mu = self.mu_conv(h, adj)
        if self.training:
            logstd = self.logstd_conv(h, adj)
            std = torch.exp(logstd)
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mu)
            return z, mu, logstd
        else:
            return mu
