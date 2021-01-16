import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch import optim

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.layers.pytorch import GraphConvolution
from graphgallery.nn.metrics.pytorch import Accuracy
from graphgallery.nn.init.pytorch import glorot_uniform, zeros


class SimPGCN(TorchKeras):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hids=[64],
                 acts=[None],
                 lambda_=5.0,
                 gamma=0.1,
                 dropout=0.5,
                 weight_decay=5e-4,
                 lr=0.01,
                 use_bias=False):

        super().__init__()
        self.lambda_ = lambda_
        self.gamma = gamma

        layers = nn.ModuleList()
        inc = in_channels
        for hid, act in zip(hids, acts):
            layer = GraphConvolution(inc,
                                     hid,
                                     activation=act,
                                     use_bias=use_bias)
            layers.append(layer)
            inc = hid

        layer = GraphConvolution(inc,
                                 out_channels,
                                 use_bias=use_bias)
        layers.append(layer)

        self.layers = layers
        self.scores = nn.ParameterList()
        self.bias = nn.ParameterList()
        self.D_k = nn.ParameterList()
        self.D_bias = nn.ParameterList()

        for hid in [in_channels] + hids:
            self.scores.append(nn.Parameter(torch.FloatTensor(hid, 1)))
            self.bias.append(nn.Parameter(torch.FloatTensor(1)))
            self.D_k.append(nn.Parameter(torch.FloatTensor(hid, 1)))
            self.D_bias.append(nn.Parameter(torch.FloatTensor(1)))
            

        # discriminator for ssl
        self.linear = nn.Linear(hids[-1], 1)

        self.compile(loss=torch.nn.CrossEntropyLoss(),
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
            b.data.fill_(0.)

        for Dk in self.D_k:
            glorot_uniform(Dk)

        for b in self.D_bias:
            b.data.fill_(0.)

    def forward(self, x, adj, adj_knn=None):

        adj_knn = self.from_cache(adj_knn=adj_knn)
        gamma = self.gamma
        embeddings = None
        
        for ix, layer in enumerate(self.layers):
            s = torch.sigmoid(x @ self.scores[ix] + self.bias[ix])
            Dk = x @ self.D_k[ix] + self.D_bias[ix]
            x = s * layer(x, adj) + (1 - s) * layer(x, adj_knn) + gamma * Dk * layer(x)
            
            if ix < len(self.layers) - 1:
                x = self.dropout(x)

            if ix == len(self.layers) - 2:
                embeddings = x.clone()

        # self.ss = torch.cat((s_i.view(1, -1), s_o.view(1, -1), gamma * Dk_i.view(1, -1), gamma * Dk_o.view(1, -1)), dim=0)

        if self.training:
            return x, embeddings
        else:
            return x

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

    def train_step_on_batch(self,
                            x,
                            y=None,
                            out_weight=None,
                            device="cpu"):
        self.train()
        optimizer = self.optimizer
        loss_fn = self.loss
        metrics = self.metrics
        optimizer.zero_grad()

        assert len(x) == 5
        *x, pseudo_labels, node_pairs = x

        out, embeddings = self(*x)
        if out_weight is not None:
            out = out[out_weight]

        # TODO
        loss = loss_fn(out, y) + self.lambda_ * self.regression_loss(embeddings, pseudo_labels, node_pairs)

        loss.backward()
        optimizer.step()
        for metric in metrics:
            metric.update_state(y.cpu(), out.detach().cpu())

        results = [loss.cpu().detach()] + [metric.result() for metric in metrics]
        return dict(zip(self.metrics_names, results))
