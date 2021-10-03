import torch
import torch.nn as nn
from torch import optim

from graphgallery.nn.models import TorchEngine
from graphgallery.nn.models.pytorch.graphat.utils import *
from graphgallery.nn.layers.pytorch import GCNConv, Sequential, activations
from graphgallery.nn.metrics.pytorch import Accuracy


class GCNVAT(TorchEngine):
    def __init__(self,
                 in_features,
                 out_features,
                 *,
                 xi=1e-4,
                 alpha=1.0,
                 epsilon=1.0,
                 num_power_iterations=1,
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
        self.xi = xi
        self.alpha = alpha
        self.epsilon = epsilon
        self.num_power_iterations = num_power_iterations

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
            loss += self.alpha * self.virtual_adversarial_loss(x, z)
        return loss

    def generate_virtual_adversarial_perturbation(self, inputs, logit):
        x, adj = inputs
        d = nn.Parameter(torch.randn_like(x))
        for _ in range(self.num_power_iterations):
            d = self.xi * d / torch.norm(d, p=2, dim=1, keepdim=True)
            logit_p = logit
            logit_m = self(x + d, adj)
            dist = kld_with_logits(logit_p, logit_m)
            d = torch.autograd.grad(dist, d)[0].detach()

        return self.epsilon * d / torch.norm(d, p=2, dim=1, keepdim=True)

    def virtual_adversarial_loss(self, inputs, logit):
        x, adj = inputs
        r_adv = self.generate_virtual_adversarial_perturbation(inputs, logit)
        logit_p = logit.detach()
        logit_q = self(x + r_adv, adj)
        return kld_with_logits(logit_p, logit_q)
