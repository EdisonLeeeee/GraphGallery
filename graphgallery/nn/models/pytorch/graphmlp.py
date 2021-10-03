import torch
import torch.nn as nn
from torch import optim

from graphgallery.nn.models.torch_engine import TorchEngine, to_device
from graphgallery.nn.layers.pytorch import Sequential, activations
from graphgallery.nn.metrics.pytorch import Accuracy


class MLP(nn.Module):
    def __init__(self, in_features, out_features, act='gelu', dropout=0.6, bias=True):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features, bias=bias)
        self.fc2 = nn.Linear(out_features, out_features, bias=bias)
        self.act = activations.get(act)

        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(out_features, eps=1e-6)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.layernorm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_feature_dis(x):
    """
    x :           batch_size x nhid
    x_dis(i,j):   item means the similarity between x(i) and x(j).
    """
    x_dis = x @ x.T
    mask = torch.eye(x_dis.size(0), device=x.device)
    x_sum = torch.sum(x**2, 1).view(-1, 1)
    x_sum = torch.sqrt(x_sum).view(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis * (x_sum**(-1))
    x_dis = (1 - mask) * x_dis
    return x_dis


def Ncontrast(x_dis, adj_label, tau=1):
    """
    compute the Ncontrast loss
    """
    x_dis = torch.exp(tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis * adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))).mean()
    return loss


class GraphMLP(TorchEngine):
    def __init__(self,
                 in_features,
                 out_features,
                 *,
                 tau=2,
                 alpha=10.0,
                 hids=[256],
                 acts=['gelu'],
                 dropout=0.6,
                 weight_decay=5e-3,
                 lr=0.001,
                 bias=True):

        super().__init__()
        mlp = []
        for hid, act in zip(hids, acts):
            mlp.append(MLP(in_features, hid, act=act, dropout=dropout, bias=bias))
            in_features = hid
        self.mlp = Sequential(*mlp)
        self.classifier = nn.Linear(in_features, out_features, bias=bias)
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam(self.parameters(),
                                          weight_decay=weight_decay,
                                          lr=lr),
                     metrics=[Accuracy()])
        self.tau = tau
        self.alpha = alpha

    def forward(self, x):
        h = self.mlp(x)
        z = self.classifier(h)
        return dict(z=z, h=h)

    def compute_loss(self, output_dict, y):
        pred = output_dict['pred']

        if self.training:
            x_dis = get_feature_dis(output_dict['h'])
            loss = self.loss(pred, y[0]) + Ncontrast(x_dis, y[1], tau=self.tau) * self.alpha
        else:
            loss = self.loss(pred, y)
        return loss
