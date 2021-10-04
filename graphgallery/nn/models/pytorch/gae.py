import torch
import torch.nn as nn
from torch import optim

from graphgallery.nn.layers.pytorch import GCNConv, Sequential, activations, InnerProductDecoder
from graphgallery.nn.metrics.pytorch import AveragePrecision, AUC
from graphgallery.nn.losses import BCELoss
from graphgallery.nn.models import TorchEngine

from graphgallery.functional.torch_utils import negative_sampling


class GAE(TorchEngine):
    def __init__(self,
                 in_features,
                 *,
                 out_features=16,
                 hids=[32],
                 acts=['relu'],
                 dropout=0.,
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
        self.compile(loss=BCELoss(),
                     optimizer=optim.Adam(self.parameters(),
                                          weight_decay=weight_decay, lr=lr),
                     metrics=[AUC(), AveragePrecision()])

    def forward(self, x, adj):
        z = self.encoder(x, adj)
        return z

    def compute_loss(self, output_dict, y):
        if self.training:
            y = output_dict['y']
        loss = self.loss(output_dict['pred'], y)
        return loss


class VGAE(TorchEngine):
    def __init__(self,
                 in_features,
                 *,
                 out_features=16,
                 hids=[32],
                 acts=['relu'],
                 dropout=0.,
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

        self.mu_conv = GCNConv(in_features, out_features, bias=bias)
        self.logstd_conv = GCNConv(in_features, out_features, bias=bias)
        self.conv = Sequential(*conv)
        self.decoder = InnerProductDecoder()
        self.compile(loss=BCELoss(),
                     optimizer=optim.Adam(self.parameters(),
                                          weight_decay=weight_decay, lr=lr),
                     metrics=[AUC(), AveragePrecision()])

    def forward(self, x, adj):
        h = self.conv(x, adj)
        mu = self.mu_conv(h, adj)
        if self.training:
            logstd = self.logstd_conv(h, adj)
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


def forward_step(self, x, out_index=None):
    # the input `x` can be: (1) dict (2) list or tuple like
    if isinstance(x, dict):
        output_dict = self(**x)
    else:
        if not isinstance(x, (list, tuple)):
            x = (x,)
        output_dict = self(*x)

    if not isinstance(output_dict, dict):
        if isinstance(output_dict, tuple):
            raise RuntimeError("For model more than 1 outputs, we recommend you to use dict as returns.")
        # Here `z` is the final representation of the model
        z = output_dict
        output_dict = dict(z=z)
    else:
        z = output_dict['z']

    if not self.training:
        output_dict['pred'] = self.decoder(z, out_index)
        return output_dict

    # here `out_index` maybe pos_edge_index
    # or (pos_edge_index, neg_edge_index)
    if isinstance(out_index, (list, tuple)):
        assert len(out_index) == 2, '`out_index` should be (pos_edge_index, neg_edge_index) or pos_edge_index'
        pos_edge_index, neg_edge_index = out_index
    else:
        pos_edge_index, neg_edge_index = out_index, None
    pos_pred = self.decoder(z, pos_edge_index)
    pos_y = z.new_ones(pos_edge_index.size(1))

    if neg_edge_index is None:
        neg_edge_index = negative_sampling(pos_edge_index, z.size(0))

    neg_pred = self.decoder(z, neg_edge_index)
    neg_y = z.new_zeros(neg_edge_index.size(1))
    pred = torch.cat([pos_pred, neg_pred], dim=0)
    y = torch.cat([pos_y, neg_y], dim=0)

    output_dict['pos_pred'] = pos_pred
    output_dict['neg_pred'] = neg_pred
    output_dict['pred'] = pred
    output_dict['y'] = y
    return output_dict


GAE.forward_step = forward_step
VGAE.forward_step = forward_step
