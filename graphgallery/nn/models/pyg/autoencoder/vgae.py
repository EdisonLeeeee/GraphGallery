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
            return z
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
