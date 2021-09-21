import torch
import torch.nn as nn
from torch import optim

from graphgallery.nn.layers.pytorch import Sequential, activations, Tree, MultiSequential
from graphgallery.nn.metrics.pytorch import AveragePrecision, AUC
from graphgallery.nn.models.pytorch.autoencoder.decoder import InnerProductDecoder

from torch_geometric.nn import GCNConv

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
                 in_features, *,
                 out_features=16,
                 hids=[32],
                 acts=['relu'],
                 dropout=0.5,
                 weight_decay=0.,
                 lr=0.01,
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

        mu = GCNConv(in_features, out_features, bias=bias)
        logstd = GCNConv(in_features, out_features, bias=bias)
        self.encoder = MultiSequential(Sequential(*conv, Tree(mu, logstd)), Reparameterize())

        self.decoder = InnerProductDecoder()
        self.compile(loss=BCELoss(),
                     optimizer=optim.Adam(self.parameters(),
                                          weight_decay=weight_decay, lr=lr),
                     metrics=[AUC(), AveragePrecision()])

    def forward(self, x, edge_index, edge_weight=None):
        z = self.encode(x, edge_index, edge_weight)
        return z

    def encode(self, x, edge_index, edge_weight=None):
        if self.training:
            z, mu, logstd = self.encoder(x, edge_index, edge_weight)
            self.cache['mu'] = mu
            self.cache['logstd'] = logstd

        else:
            z = self.encoder(x, edge_index, edge_weight)
        return z

    def compute_loss(self, pos_pred, neg_pred):
        mu = self.cache.pop('mu')
        logstd = self.cache.pop('logstd')
        kl_loss = -0.5 / mu.size(0) * torch.mean(torch.sum(1 + 2 * logstd - mu.pow(2) - logstd.exp().pow(2), dim=1))
        return self.loss(pos_pred, neg_pred) + kl_loss
