import torch
import torch.nn as nn
from torch import optim

from graphgallery.nn.models.torch_keras import TorchKeras, to_device
from graphgallery.nn.models.pytorch.graphat.utils import *
from graphgallery.nn.layers.pytorch import GCNConv, Sequential, activations
from graphgallery.nn.metrics.pytorch import Accuracy


class GCN_VAT(TorchKeras):
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
                     optimizer=optim.Adam([dict(params=conv[1].parameters(),
                                                weight_decay=weight_decay),
                                           dict(params=conv[2:].parameters(),
                                                weight_decay=0.)], lr=lr),
                     metrics=[Accuracy()])
        self.xi = xi
        self.alpha = alpha
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
        optimizer.zero_grad()
        x, y = to_device(x, y, device=device)
        logit = self(*x)
        out = self.index_select(logit, out_index=out_index)
        loss = loss_fn(out, y) + self.alpha * self.virtual_adversarial_loss(x, logit)

        loss.backward()
        optimizer.step()
        self.update_metrics(out, y)

        results = [loss.cpu().detach()] + [metric.result() for metric in self.metrics]
        return dict(zip(self.metrics_names, results))

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
