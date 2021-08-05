import torch.nn as nn
from torch import optim

from graphgallery.nn.layers.pytorch import Sequential, activations
from graphgallery.nn.metrics.pytorch import AveragePrecision, AUC
from graphgallery.nn.models.pytorch.autoencoder.decoder import InnerProductDecoder

from torch_geometric.nn import GCNConv

from .autoencoder import AutoEncoder
from .loss import LogLoss


class GAE(AutoEncoder):
    def __init__(self,
                 in_features, *,
                 out_features=16,
                 hids=[32],
                 acts=['relu'],
                 dropout=0.5,
                 weight_decay=0.,
                 lr=0.01,
                 bias=True):
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
        self.compile(loss=LogLoss(),
                     optimizer=optim.Adam(self.parameters(),
                                          weight_decay=weight_decay, lr=lr),
                     metrics=[AUC(), AveragePrecision()])

    def forward(self, x, edge_index, edge_weight=None):
        z = self.encode(x, edge_index, edge_weight)
        return z
