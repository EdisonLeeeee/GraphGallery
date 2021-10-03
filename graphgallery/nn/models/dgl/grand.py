import torch
import numpy as np
import dgl.function as fn

import torch.nn as nn
from torch import optim

from graphgallery.nn.models import TorchEngine
from graphgallery.nn.layers.pytorch import activations
from graphgallery.nn.metrics.pytorch import Accuracy


def drop_node(feats, drop_rate, training):

    n = feats.shape[0]
    drop_rates = torch.ones(n) * drop_rate

    if training:

        masks = torch.bernoulli(1. - drop_rates).unsqueeze(1)
        feats = masks.to(feats.device) * feats

    else:
        feats = feats * (1. - drop_rate)

    return feats


def GRANDConv(graph, feats, order):
    '''
    Parameters
    -----------
    graph: dgl.Graph
        The input graph
    feats: Tensor (n_nodes * feat_dim)
        Node features
    order: int
        Propagation Steps
    '''
    with graph.local_scope():

        ''' Calculate Symmetric normalized adjacency matrix   \hat{A} '''
        degs = graph.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -0.5).to(feats.device).unsqueeze(1)

        graph.ndata['norm'] = norm
        graph.apply_edges(fn.u_mul_v('norm', 'norm', 'weight'))

        ''' Graph Conv '''
        x = feats
        y = 0 + feats

        for i in range(order):
            graph.ndata['h'] = x
            graph.update_all(fn.u_mul_e('h', 'weight', 'm'), fn.sum('m', 'h'))
            x = graph.ndata.pop('h')
            y.add_(x)

    return y / (order + 1)


class GRAND(TorchEngine):
    def __init__(self,
                 in_features,
                 out_features,
                 hids=[16],
                 acts=['relu'],
                 dropout=0.5,
                 S=1,
                 K=4,
                 temp=0.5,
                 lam=1.,
                 weight_decay=5e-4,
                 lr=0.01,
                 bias=False,
                 bn=False):

        super().__init__()

        mlp = []
        for hid, act in zip(hids, acts):
            if bn:
                mlp.append(nn.BatchNorm1d(in_features))
            mlp.append(nn.Linear(in_features,
                                 hid,
                                 bias=bias))
            mlp.append(activations.get(act))
            mlp.append(nn.Dropout(dropout))
            in_features = hid
        if bn:
            mlp.append(nn.BatchNorm1d(in_features))
        mlp.append(nn.Linear(in_features, out_features, bias=bias))
        self.mlp = mlp = nn.Sequential(*mlp)
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam(mlp.parameters(),
                                          weight_decay=weight_decay, lr=lr),
                     metrics=[Accuracy()])
        self.S = S
        self.K = K
        self.temp = temp
        self.lam = lam
        self.dropout = dropout

    def forward(self, feats, graph):
        X = feats
        S = self.S

        if self.training:  # Training Mode
            output_list = []
            for _ in range(S):
                drop_feat = drop_node(X, self.dropout, True)  # Drop node
                feat = GRANDConv(graph, drop_feat, self.K)    # Graph Convolution
                output_list.append(self.mlp(feat))  # Prediction

            return output_list
        else:   # Inference Mode
            drop_feat = drop_node(X, self.dropout, False)
            X = GRANDConv(graph, drop_feat, self.K)

            return self.mlp(X)

    def compute_loss(self, outs, y, out_index=None):
        if self.training:
            loss_consis = consis_loss(outs, temp=self.temp, lam=self.lam)
            outs = [self.index_select(out, out_index=out_index) for out in outs]
            loss_sup = 0.
            for out in outs:
                loss_sup += self.loss(out, y)
            loss_sup = loss_sup / len(outs)
            return loss_sup + loss_consis, outs[0]
        else:
            out = self.index_select(outs, out_index=out_index)
            return self.loss(out, y), out


def consis_loss(logps, temp=0.5, lam=1.):
    ps = [torch.exp(p) for p in logps]
    ps = torch.stack(ps, dim=2)

    avg_p = torch.mean(ps, dim=2)
    sharp_p = (torch.pow(avg_p, 1. / temp) / torch.sum(torch.pow(avg_p, 1. / temp), dim=1, keepdim=True)).detach()

    sharp_p = sharp_p.unsqueeze(2)
    loss = torch.mean(torch.sum(torch.pow(ps - sharp_p, 2), dim=1, keepdim=True))

    loss = lam * loss
    return loss
