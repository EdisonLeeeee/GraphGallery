import torch.nn as nn
from graphgallery.nn.layers.pytorch import Sequential, activations


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


class GraphMLP(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 *,
                 hids=[256],
                 acts=['gelu'],
                 dropout=0.6,
                 bias=True):

        super().__init__()
        mlp = []
        for hid, act in zip(hids, acts):
            mlp.append(MLP(in_features, hid, act=act,
                           dropout=dropout, bias=bias))
            in_features = hid
        self.mlp = Sequential(*mlp)
        self.classifier = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        h = self.mlp(x)
        z = self.classifier(h)
        if self.training:
            return z, h
        else:
            return z
