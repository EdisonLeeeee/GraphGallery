import torch
import torch.nn as nn

from graphgallery.nn.layers.pytorch import GCNConv, activations
from graphgallery.nn.init import glorot_uniform, zeros


class SimPGCN(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 hids=[64],
                 acts=[None],
                 gamma=0.1,
                 dropout=0.5,
                 bias=False):

        super().__init__()
        self.gamma = gamma
        assert hids, "hids should not empty"
        layers = nn.ModuleList()
        act_layers = nn.ModuleList()
        inc = in_features
        for hid, act in zip(hids, acts):
            layers.append(GCNConv(in_features,
                                  hid,
                                  bias=bias))
            act_layers.append(activations.get(act))
            inc = hid

        layers.append(GCNConv(inc,
                              out_features,
                              bias=bias))
        act_layers.append(activations.get(None))

        self.layers = layers
        self.act_layers = act_layers
        self.scores = nn.ParameterList()
        self.bias = nn.ParameterList()
        self.D_k = nn.ParameterList()
        self.D_bias = nn.ParameterList()

        for hid in [in_features] + hids:
            self.scores.append(nn.Parameter(torch.FloatTensor(hid, 1)))
            self.bias.append(nn.Parameter(torch.FloatTensor(1)))
            self.D_k.append(nn.Parameter(torch.FloatTensor(hid, 1)))
            self.D_bias.append(nn.Parameter(torch.FloatTensor(1)))

        # discriminator for ssl
        self.linear = nn.Linear(hids[-1], 1)
        self.dropout = nn.Dropout(dropout)
        self._adj_knn = None
        self.reset_parameters()

    def reset_parameters(self):

        for layer in self.layers:
            layer.reset_parameters()

        for s in self.scores:
            glorot_uniform(s)

        for b in self.bias:
            # fill in b with positive value to make
            # score s closer to 1 at the beginning
            zeros(b)

        for Dk in self.D_k:
            glorot_uniform(Dk)

        for b in self.D_bias:
            zeros(b)

    def forward(self, x, adj, adj_knn=None):

        if adj_knn is None:
            adj_knn = self._adj_knn
        else:
            self._adj_knn = adj_knn

        gamma = self.gamma
        h = None

        for ix, (layer, act) in enumerate(zip(self.layers, self.act_layers)):
            s = torch.sigmoid(x @ self.scores[ix] + self.bias[ix])
            Dk = x @ self.D_k[ix] + self.D_bias[ix]
            x = s * act(layer(x, adj)) + (1 - s) * \
                act(layer(x, adj_knn)) + gamma * Dk * act(layer(x))

            if ix < len(self.layers) - 1:
                x = self.dropout(x)

            if ix == len(self.layers) - 2:
                h = x.clone()

        z = x
        # self.ss = torch.cat((s_i.view(1, -1), s_o.view(1, -1), gamma * Dk_i.view(1, -1), gamma * Dk_o.view(1, -1)), dim=0)
        if self.training:
            return z, h
        else:
            return z

    def cache_clear(self):
        self._adj_knn = None
        return self
