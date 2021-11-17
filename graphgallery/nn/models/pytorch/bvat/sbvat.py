import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from graphgallery.nn.models.pytorch.bvat.utils import *
from graphgallery.nn.layers.pytorch import GCNConv, Sequential, activations
from graphgallery.nn.metrics import Accuracy


class SBVAT(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 *,
                 xi=1e-6,
                 p1=1.0,
                 p2=1.0,
                 epsilon=5e-2,
                 num_power_iterations=1,
                 hids=[16],
                 acts=['relu'],
                 dropout=0.5,
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
                     optimizer=optim.Adam([dict(params=conv[1].parameters(),
                                                weight_decay=weight_decay),
                                           dict(params=conv[2:].parameters(),
                                                weight_decay=0.)], lr=lr),
                     metrics=[Accuracy()])
        self.xi = xi
        self.p1 = p1
        self.p2 = p2
        self.epsilon = epsilon
        self.num_power_iterations = num_power_iterations

    def forward(self, x, adj):
        return self.conv(x, adj)

    def forward_step(self, x, out_index=None):
        if self.training:
            z = self(*x[:-1])
        else:
            z = self(*x)

        pred = self.index_select(z, out_index=out_index)
        return dict(z=z, x=x, pred=pred)

    def compute_loss(self, output_dict, y):
        # index select or mask outputs
        z = output_dict['z']
        pred = output_dict['pred']
        loss = self.loss(pred, y)

        if self.training:
            x = output_dict['x']
            loss += + self.p1 * self.virtual_adversarial_loss(x, z) + \
                self.p2 * self.entropy_loss(z)
        return loss

    def generate_virtual_adversarial_perturbation(self, inputs, logit):
        x, adj, adv_mask = inputs
        d = nn.Parameter(torch.randn_like(x))
        for _ in range(self.num_power_iterations):
            d = self.xi * l2_normalize(d)
            logit_p = logit
            logit_m = self(x + d, adj)
            dist = masked_kld_with_logits(logit_p, logit_m, adv_mask)
            d = torch.autograd.grad(dist, d)[0].detach()
        return self.epsilon * l2_normalize(d)

    def virtual_adversarial_loss(self, inputs, logit):
        x, adj, adv_mask = inputs
        r_adv = self.generate_virtual_adversarial_perturbation(inputs, logit)
        logit_p = logit.detach()
        logit_q = self(x + r_adv, adj)
        return masked_kld_with_logits(logit_p, logit_q, adv_mask)

    def entropy_loss(self, logit):
        q = F.softmax(logit, dim=-1)
        cross_entropy = softmax_cross_entropy_with_logits(logits=logit, labels=q)
        return cross_entropy.mean()
