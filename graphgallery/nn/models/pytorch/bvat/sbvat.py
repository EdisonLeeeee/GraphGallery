import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.models.pytorch.bvat.utils import *
from graphgallery.nn.layers.pytorch import GCNConv, Sequential, activations
from graphgallery.nn.metrics.pytorch import Accuracy


class SBVAT(TorchKeras):
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

    def train_step_on_batch(self,
                            x,
                            y,
                            out_index=None,
                            device="cpu"):
        self.train()
        optimizer = self.optimizer
        loss_fn = self.loss
        metrics = self.metrics
        optimizer.zero_grad()
        x = [_x.to(device) if hasattr(_x, 'to') else _x for _x in x]
        y = y.to(device)
        logit = self(*x[:-1])
        out = logit
        if out_index is not None:
            out = logit[out_index]
        loss = loss_fn(out, y) + self.p1 * self.virtual_adversarial_loss(x, logit) + \
            self.p2 * self.entropy_loss(logit)

        loss.backward()
        optimizer.step()
        for metric in metrics:
            metric.update_state(y.cpu(), out.detach().cpu())

        results = [loss.cpu().detach()] + [metric.result() for metric in metrics]
        return dict(zip(self.metrics_names, results))

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
