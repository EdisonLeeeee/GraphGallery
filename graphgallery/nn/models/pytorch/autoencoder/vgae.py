import torch
import torch.nn as nn
from torch import optim

from graphgallery.nn.layers.pytorch import GCNConv, Sequential, Tree, MultiSequential, activations
from graphgallery.nn.metrics.pytorch import AveragePrecision, AUC
from .decoder import InnerProductDecoder
from .autoencoder import AutoEncoder
from .loss import BCELoss


class Reparameterize(nn.Module):

    def forward(self, mu, logstd):
        if self.training:
            std = torch.exp(logstd)
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mu)
            return z, mu, logstd
        else:
            return mu


class VGAE(AutoEncoder):
    def __init__(self,
                 in_features,
                 *,
                 out_features=16,
                 hids=[32],
                 acts=['relu'],
                 dropout=0.,
                 pos_weight=1.0,
                 weight_decay=0.,
                 lr=0.01,
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
        mu = GCNConv(in_features, out_features, bias=bias)
        logstd = GCNConv(in_features, out_features, bias=bias)

        self.encoder = MultiSequential(Sequential(*conv, Tree(mu, logstd)), Reparameterize())
        self.decoder = InnerProductDecoder()
        self.compile(loss=BCELoss(pos_weight=pos_weight),
                     optimizer=optim.Adam(self.parameters(),
                                          weight_decay=weight_decay, lr=lr),
                     metrics=[AUC(), AveragePrecision()])

    def forward(self, x, adj):
        if self.training:
            z, mu, logstd = self.encode(x, adj)
            self.cache['mu'] = mu
            self.cache['logstd'] = logstd
        else:
            z = self.encode(x, adj)
        out = self.decode(z)
        return out

    def compute_loss(self, out, y, out_index=None):
        out = self.index_select(out, out_index=out_index)
        if self.training:
            mu = self.cache.pop('mu')
            logstd = self.cache.pop('logstd')
            kl_loss = -0.5 / mu.size(0) * torch.mean(torch.sum(1 + 2 * logstd - mu.pow(2) - logstd.exp().pow(2), dim=1))
        else:
            kl_loss = 0.
        return self.loss(out, y) + kl_loss, out
