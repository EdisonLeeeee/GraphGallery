import torch
import torch.nn as nn
from graphgallery.nn.layers.pytorch import activations


class Node2GridsCNN(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 mapsize_a,
                 mapsize_b,
                 conv_channel=64,
                 hids=[200],
                 acts=['relu6'],
                 attnum=10,
                 dropout=0.6,
                 bias=True):

        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_features,
                out_channels=conv_channel,
                kernel_size=(2, 1),
                stride=1,
                padding=0
            ),
            nn.Softmax(dim=1),
        )
        lin = []
        in_features = (mapsize_a - 1) * mapsize_b * conv_channel
        for hid, act in zip(hids, acts):
            lin.append(nn.Linear(in_features, hid, bias=bias))
            lin.append(activations.get(act))
            lin.append(nn.Dropout(dropout))
            in_features = hid
        lin.append(nn.Linear(in_features, out_features, bias=bias))

        self.lin = nn.Sequential(*lin)
        self.attention = nn.Parameter(torch.ones(attnum, mapsize_a - 1, mapsize_b))

    def forward(self, x):
        attention = torch.sum(self.attention, dim=0) / self.attention.size(0)
        x = self.conv(x)
        x = attention * x + x
        x = x.view(x.size(0), -1)
        out = self.lin(x)
        return out
