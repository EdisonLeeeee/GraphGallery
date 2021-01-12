import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch import optim

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.layers.pytorch import GraphConvolution
from graphgallery.nn.metrics.pytorch import Accuracy
from graphgallery.nn.init.pytorch import glorot_uniform


class SimPGCN(TorchKeras):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hids=[64],
                 acts=['relu'],
                 lambda_=5.0,
                 gamma=0.1,
                 dropout=0.5,
                 weight_decay=5e-4,
                 lr=0.01,
                 use_bias=True):

        super().__init__()
        self.lambda_ = lambda_
        self.gamma = gamma

        # TODO: more layers
        assert len(hids) == 1
        
        nhid = hids[0]
        self.gc1 = GraphConvolution(in_channels, nhid, use_bias=use_bias)
        self.gc2 = GraphConvolution(nhid, out_channels, use_bias=use_bias)

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
        self.linear = nn.Linear(nhid, 1)

        self.compile(loss=torch.nn.CrossEntropyLoss(),
                     optimizer=optim.Adam(self.parameters(), lr=lr,
                                          weight_decay=weight_decay),
                     metrics=[Accuracy()])
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):

        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

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

        s_i = torch.sigmoid(x @ self.scores[0] + self.bias[0])
        Dk_i = (x @ self.D_k[0] + self.D_bias[0])
        x = (s_i * self.gc1(x, adj) + (1 - s_i) * self.gc1(x, adj_knn)) + (gamma) * Dk_i * self.gc1(x)

        x = self.dropout(x)
        embedding = x.clone()

        # output, no relu and dropput here.
        s_o = torch.sigmoid(x @ self.scores[-1] + self.bias[-1])
        Dk_o = (x @ self.D_k[-1] + self.D_bias[-1])
        x = (s_o * self.gc2(x, adj) + (1 - s_o) * self.gc2(x, adj_knn)) + (gamma) * Dk_o * self.gc2(x)

        self.ss = torch.cat((s_i.view(1, -1), s_o.view(1, -1), gamma * Dk_i.view(1, -1), gamma * Dk_o.view(1, -1)), dim=0)
        
        if self.training:
            return x, embedding
        else:
            return x

    def regression_loss(self, embeddings, pseudo_labels=None, node_pairs=None):
        pseudo_labels,  node_pairs = self.from_cache(pseudo_labels=pseudo_labels, 
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
