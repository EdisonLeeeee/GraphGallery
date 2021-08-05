import torch
import torch.nn as nn
from torch import optim

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.layers.pytorch import GCNConv, Sequential, activations
from graphgallery.nn.metrics.pytorch import Accuracy


class LATGCN(TorchKeras):
    def __init__(self,
                 in_features,
                 out_features,
                 num_nodes,
                 *,
                 gamma=0.01,
                 eta=0.1,
                 hids=[16],
                 acts=['relu'],
                 dropout=0.2,
                 weight_decay=5e-4,
                 lr=0.01,
                 bias=False):
        super().__init__()
        assert hids, "LATGCN requires hidden layers"
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

        self.zeta = nn.Parameter(torch.randn(num_nodes, hids[0]))
        self.conv1 = conv[:3]  # includes dropout, ReLU and the first GCN layer
        self.conv2 = conv[3:]  # remainder
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam([dict(params=self.conv1.parameters(),
                                                weight_decay=weight_decay),
                                           dict(params=self.conv2.parameters(),
                                                weight_decay=0.)], lr=lr),
                     metrics=[Accuracy()])

        self.zeta_opt = optim.Adam([self.zeta], lr=lr)

        self.gamma = gamma
        self.eta = eta

    def forward(self, x, adj):
        h = self.conv1(x, adj)
        logit = self.conv2(h, adj)

        if self.training:
            self.zeta.data = clip_by_norm(self.zeta, self.eta)
            hp = h + self.zeta
            logitp = self.conv2(hp, adj)
            reg_loss = torch.norm(logitp - logit)
            return logit, reg_loss
        else:
            return logit

    def train_step_on_batch(self,
                            x,
                            y,
                            out_index=None,
                            device="cpu"):
        self.train()
        optimizer = self.optimizer
        loss_fn = self.loss
        metrics = self.metrics
        if not isinstance(x, (list, tuple)):
            x = [x]
        x = [_x.to(device) if hasattr(_x, 'to') else _x for _x in x]
        y = y.to(device)

        zeta_opt = self.zeta_opt
        for _ in range(20):
            zeta_opt.zero_grad()
            _, reg_loss = self(*x)
            reg_loss = -reg_loss
            reg_loss.backward()
            zeta_opt.step()

        optimizer.zero_grad()
        out, reg_loss = self(*x)
        if out_index is not None:
            out = out[out_index]
        loss = loss_fn(out, y) + self.gamma * reg_loss
        loss.backward()
        optimizer.step()
        for metric in metrics:
            metric.update_state(y.cpu(), out.detach().cpu())

        results = [loss.cpu().detach()] + [metric.result() for metric in metrics]
        return dict(zip(self.metrics_names, results))


@torch.no_grad()
def clip_by_norm(tensor, clip_norm):
    l2_norm = torch.norm(tensor, p=2, dim=1).view(-1, 1)
    tensor = tensor * clip_norm / l2_norm
    return tensor
