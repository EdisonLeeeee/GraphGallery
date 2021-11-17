import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch import optim

from graphgallery.nn.models import TorchEngine
from graphgallery.nn.layers.pytorch import GCNConv, activations
from graphgallery.nn.metrics import Accuracy
from graphgallery.nn.init import glorot_uniform, zeros


class SimPGCN(TorchEngine):
    def __init__(self,
                 in_features,
                 out_features,
                 hids=[64],
                 acts=[None],
                 lambda_=5.0,
                 gamma=0.1,
                 dropout=0.5,
                 weight_decay=5e-4,
                 lr=0.01,
                 bias=False):

        super().__init__()
        self.lambda_ = lambda_
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

        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay),
                     metrics=[Accuracy()])

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):

        for layer in self.layers:
            layer.reset_parameters()

        for s in self.scores:
            glorot_uniform(s)

        for b in self.bias:
            # fill in b with postive value to make
            # score s closer to 1 at the beginning
            zeros(b)

        for Dk in self.D_k:
            glorot_uniform(Dk)

        for b in self.D_bias:
            zeros(b)

    def forward(self, x, adj, adj_knn=None):

        adj_knn = self.from_cache(adj_knn=adj_knn)
        gamma = self.gamma
        h = None

        for ix, (layer, act) in enumerate(zip(self.layers, self.act_layers)):
            s = torch.sigmoid(x @ self.scores[ix] + self.bias[ix])
            Dk = x @ self.D_k[ix] + self.D_bias[ix]
            x = s * act(layer(x, adj)) + (1 - s) * act(layer(x, adj_knn)) + gamma * Dk * act(layer(x))

            if ix < len(self.layers) - 1:
                x = self.dropout(x)

            if ix == len(self.layers) - 2:
                h = x.clone()

        z = x
        # self.ss = torch.cat((s_i.view(1, -1), s_o.view(1, -1), gamma * Dk_i.view(1, -1), gamma * Dk_o.view(1, -1)), dim=0)
        return dict(z=z, h=h)

    def compute_loss(self, output_dict, y):
        pred = output_dict['pred']

        if self.training:
            embeddings = output_dict['h']
            y, pseudo_labels, node_pairs = y
            loss = self.loss(pred, y) + self.lambda_ * self.regression_loss(embeddings, pseudo_labels, node_pairs)
        else:
            loss = self.loss(pred, y)
        return loss

    def regression_loss(self, embeddings, pseudo_labels=None, node_pairs=None):
        pseudo_labels, node_pairs = self.from_cache(pseudo_labels=pseudo_labels,
                                                    node_pairs=node_pairs)
        k = 10000
        if len(node_pairs[0]) > k:
            sampled = np.random.choice(len(node_pairs[0]), k, replace=False)

            embeddings0 = embeddings[node_pairs[0][sampled]]
            embeddings1 = embeddings[node_pairs[1][sampled]]
            embeddings = self.linear(torch.abs(embeddings0 - embeddings1))
            loss = F.mse_loss(embeddings, pseudo_labels[sampled].unsqueeze(-1), reduction='mean')
        else:
            embeddings0 = embeddings[node_pairs[0]]
            embeddings1 = embeddings[node_pairs[1]]
            embeddings = self.linear(torch.abs(embeddings0 - embeddings1))
            loss = F.mse_loss(embeddings, pseudo_labels.unsqueeze(-1), reduction='mean')
        return loss
