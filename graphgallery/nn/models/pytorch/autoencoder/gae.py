import torch
import torch.nn as nn
from torch import optim

from graphgallery.nn.layers.pytorch import GCNConv, Sequential, activations
from graphgallery.nn.metrics.pytorch import AveragePrecision, AUC
from .decoder import InnerProductDecoder
from .autoencoder import AutoEncoder
from .loss import BCELoss


class GAE(AutoEncoder):
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
        self.compile(loss=BCELoss(pos_weight=pos_weight),
                     optimizer=optim.Adam(self.parameters(),
                                          weight_decay=weight_decay, lr=lr),
                     metrics=[AUC(), AveragePrecision()])

    def forward(self, x, adj):
        z = self.encode(x, adj)
        out = self.decode(z)
        return out
