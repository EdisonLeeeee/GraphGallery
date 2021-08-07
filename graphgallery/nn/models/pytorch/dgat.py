import torch
import torch.nn as nn
from torch import optim

from graphgallery.nn.models.torch_keras import TorchKeras, to_device
from graphgallery.nn.models.pytorch.graphat.utils import *
from graphgallery.nn.layers.pytorch import GCNConv, Sequential, activations
from graphgallery.nn.metrics.pytorch import Accuracy


class DGAT(TorchKeras):
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
                     optimizer=optim.Adam([dict(params=conv[1].parameters(),
                                                weight_decay=weight_decay),
                                           dict(params=conv[2:].parameters(),
                                                weight_decay=0.)], lr=lr),
                     metrics=[Accuracy()])
        self.alpha = alpha
        self.epsilon = epsilon

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
        out = logit

        if out_index is not None:
            out = logit[out_index]
        loss = loss_fn(out, y) + self.alpha * self.DGAT_loss(x, logit)

        loss.backward()
        optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.update_metrics(out, y)

        results = [loss.cpu().detach()] + [metric.result() for metric in self.metrics]
        return dict(zip(self.metrics_names, results))

    def DGAT_loss(self, inputs, logit):
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
