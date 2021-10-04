import torch
import torch.nn as nn
from torch import optim

from graphgallery.nn.layers.pytorch import Sequential, activations, InnerProductDecoder
from graphgallery.nn.metrics.pytorch import AveragePrecision, AUC
from graphgallery.nn.losses import BCELoss

from torch_geometric.nn import GCNConv
from graphgallery.nn.models import TorchEngine


class GAE(TorchEngine):
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
        self.compile(loss=BCELoss(),
                     optimizer=optim.Adam(self.parameters(),
                                          weight_decay=weight_decay, lr=lr),
                     metrics=[AUC(), AveragePrecision()])

    def forward(self, x, edge_index, edge_weight=None):
        z = self.encoder(x, edge_index, edge_weight)
        return z

    def compute_loss(self, output_dict, y):
        if self.training:
            y = output_dict['y']
        loss = self.loss(output_dict['pred'], y)
        return loss


class VGAE(TorchEngine):
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

        self.mu_conv = GCNConv(in_features, out_features, bias=bias)
        self.logstd_conv = GCNConv(in_features, out_features, bias=bias)
        self.conv = Sequential(*conv)

        self.decoder = InnerProductDecoder()
        self.compile(loss=BCELoss(),
                     optimizer=optim.Adam(self.parameters(),
                                          weight_decay=weight_decay, lr=lr),
                     metrics=[AUC(), AveragePrecision()])

    def forward(self, x, edge_index, edge_weight=None):
        h = self.conv(x, edge_index, edge_weight)
        mu = self.mu_conv(h, edge_index, edge_weight)
        if self.training:
            logstd = self.logstd_conv(h, edge_index, edge_weight)
            std = torch.exp(logstd)
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mu)
            return dict(z=z, mu=mu, logstd=logstd)
        else:
            return dict(z=mu)

    def compute_loss(self, output_dict, y):
        if self.training:
            mu = output_dict.pop('mu')
            logstd = output_dict.pop('logstd')
            kl_loss = -0.5 / mu.size(0) * torch.mean(torch.sum(1 + 2 * logstd - mu.pow(2) - logstd.exp().pow(2), dim=1))
            loss = self.loss(output_dict['pred'], output_dict['y']) + kl_loss
        else:
            loss = self.loss(output_dict['pred'], y)
        return loss


from graphgallery.nn.models.pytorch.gae import forward_step
GAE.forward_step = forward_step
VGAE.forward_step = forward_step
