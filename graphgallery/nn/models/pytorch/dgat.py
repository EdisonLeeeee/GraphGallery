import torch
import torch.nn as nn
from torch import optim

from graphgallery.nn.models import TorchEngine
from graphgallery.nn.models.pytorch.graphat.utils import *
from graphgallery.nn.layers.pytorch import GCNConv, Sequential, activations
from graphgallery.nn.metrics.pytorch import Accuracy


class DGAT(TorchEngine):
    def __init__(self,
                 in_features,
                 out_features,
                 *,
                 alpha=1.0,
                 epsilon=0.9,
                 hids=[16],
                 acts=['relu'],
                 dropout=0.,
                 weight_decay=5e-4,
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
        conv.append(GCNConv(in_features, out_features, bias=bias))
        conv = Sequential(*conv)

        self.conv = conv
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam(self.parameters(),
                                          weight_decay=weight_decay, lr=lr),
                     metrics=[Accuracy()])
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, x, adj):
        return self.conv(x, adj)

    def get_outputs(self, x, out_index=None):
        z = self(*x)
        pred = self.index_select(z, out_index=out_index)
        return dict(z=z, x=x, pred=pred)

    def compute_loss(self, output_dict, y):
        z = output_dict['z']
        pred = output_dict['pred']
        loss = self.loss(pred, y)

        if self.training:
            x = output_dict['x']
            loss += self.alpha * self.dgat_loss(x, z)
        return loss

    def dgat_loss(self, inputs, logit):
        x, adj = inputs
        adj_para = nn.Parameter(adj)
        x_new = self.generate_x(x, adj)

        logit_p = logit.detach()
        logit_m = self(x_new, adj_para)
        dist = kld_with_logits(logit_p, logit_m)
        grad = torch.autograd.grad(dist, adj_para, retain_graph=True)[0].detach()

        adj_new = self.epsilon * get_normalized_vector(grad * adj)
        x_new = self.generate_x(x, adj_new)

        logit_p = logit.detach()
        logit_m = self(x_new, adj)
        loss = kld_with_logits(logit_p, logit_m)
        return loss

    def generate_x(self, x, adj):
        D = torch.diag(adj.sum(dim=1))
        L = D - adj
        x_new = (torch.eye(L.size(0), device=L.device) - L) @ x
        return x_new
