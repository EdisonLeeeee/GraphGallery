from numpy.core.numeric import Inf
import torch
import torch.nn as nn
from torch import optim

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.metrics.pytorch import Accuracy
from graphgallery.nn.layers.pytorch import activations


class Node2GridsCNN(TorchKeras):
    def __init__(self,
                 in_features,
                 out_features,
                 mapsize_a,
                 mapsize_b,
                 hids=[200],
                 acts=['relu6'],
                 attnum=10,
                 dropout=0.5,
                 weight_decay=0.00015,
                 att_reg=0.07,
                 lr=0.008,
                 bias=True):

        super().__init__()
        convnum = 64
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_features,
                out_channels=convnum,
                kernel_size=(2, 1),
                stride=1,
                padding=0
            ),
            nn.Softmax(dim=1),
        )
        lin = []
        in_features = (mapsize_a - 1) * mapsize_b * convnum
        for hid, act in zip(hids, acts):
            lin.append(nn.Linear(in_features, hid, bias=bias))
            lin.append(activations.get(act))
            lin.append(nn.Dropout(dropout))
            in_features = hid
        lin.append(nn.Linear(in_features, out_features, bias=bias))

        self.lin = nn.Sequential(*lin)
        self.attention = nn.Parameter(torch.ones(attnum, mapsize_a - 1, mapsize_b))
        self.att_reg = att_reg
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.RMSprop(self.parameters(),
                                             weight_decay=weight_decay, lr=lr),
                     metrics=[Accuracy()])

    def forward(self, x):
        attention = torch.sum(self.attention, dim=0) / self.attention.size(0)
        x = self.conv(x)
        x = attention * x + x
        x = x.view(x.size(0), -1)
        out = self.lin(x)

        return out

    def compute_loss(self, out, y, out_index=None):
        # index select or mask outputs
        out = self.index_select(out, out_index=out_index)
        attention = self.attention.view(-1)
        attentionloss = self.att_reg * torch.sum(attention ** 2)
        loss = self.loss(out, y) + attentionloss
        return loss, out
