import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from graphgallery.nn.models.pytorch.bvat.utils import *
from graphgallery.nn.layers.pytorch import GCNConv, Sequential, activations
from graphgallery.nn.metrics import Accuracy


class OBVAT(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 num_nodes,
                 *,
                 p1=1.0,
                 p2=1.0,
                 hids=[16],
                 acts=['relu'],
                 dropout=0.,
                 weight_decay=5e-4,
                 lr=0.01,
                 pt_epochs=10,
                 bias=False):
        super().__init__()
        self.r_adv = nn.Parameter(torch.zeros(num_nodes, in_features))  # it is better to use zero initializer
        self.adv_optimizer = optim.Adam([self.r_adv], lr=lr / 10)

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
        self.p1 = p1
        self.p2 = p2
        self.pt_epochs = pt_epochs

    def forward(self, x, adj):
        return self.conv(x, adj)

    def pretrain(self, x):
        self.freeze(self.conv)

        optimizer = self.adv_optimizer
        r_adv = self.r_adv
        for _ in range(self.pt_epochs):
            optimizer.zero_grad()
            z = self(*x)
            rnorm = r_adv.square().sum()  # l2 loss
            loss = rnorm - self.virtual_adversarial_loss(x, z)
            loss.backward()
            optimizer.step()

        self.defrozen(self.conv)

    def forward_step(self, x, out_index=None):
        if self.training:
            self.pretrain(x)
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
            loss += self.p1 * self.virtual_adversarial_loss(x, z) + \
                self.p2 * self.entropy_loss(z)
        return loss

    def virtual_adversarial_loss(self, inputs, logit):
        x, adj = inputs
        logit_p = logit.detach()
        logit_q = self(x + self.r_adv, adj)
        return masked_kld_with_logits(logit_p, logit_q)

    def entropy_loss(self, logit):
        q = F.softmax(logit, dim=-1)
        cross_entropy = softmax_cross_entropy_with_logits(logits=logit, labels=q)
        return cross_entropy.mean()
