import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.layers.pytorch import GraphConvolution
from graphgallery.nn.metrics.pytorch import Accuracy


class SimpGCN(TorchKeras):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hids=[16],
                 acts=['relu'],
                 dropout=0.5,
                 weight_decay=5e-4,
                 lr=0.01,
                 lambda_=5.,
                 gamma=0.1,
                 use_bias=True):

        super().__init__()
        assert len(hids) == 1
        hid = hids[0]

        self.scores = nn.ParameterList()
        self.scores.append(Parameter(torch.FloatTensor(in_channels, 1)))
        for i in range(1):
            self.scores.append(Parameter(torch.FloatTensor(hid, 1)))

        self.bias = nn.ParameterList()
        self.bias.append(Parameter(torch.FloatTensor(1)))
        for i in range(1):
            self.bias.append(Parameter(torch.FloatTensor(1)))

        self.D_k = nn.ParameterList()
        self.D_k.append(Parameter(torch.FloatTensor(in_channels, 1)))
        for i in range(1):
            self.D_k.append(Parameter(torch.FloatTensor(hid, 1)))

        self.D_bias = nn.ParameterList()
        self.D_bias.append(Parameter(torch.FloatTensor(1)))
        for i in range(1):
            self.D_bias.append(Parameter(torch.FloatTensor(1)))

        self.w = nn.Linear(hid, 1)
        self.gc1 = GraphConvolution(in_channels, hid, activation=None, use_bias=use_bias)
        self.gc2 = GraphConvolution(hid, out_channels, activation=None, use_bias=use_bias)

        self.compile(loss=torch.nn.CrossEntropyLoss(),
                     optimizer=optim.Adam(self.parameters(), lr=lr),
                     metrics=[Accuracy()])
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def forward(self, x, adj, adj_knn):
        x, _ = self.myforward(x, adj, adj_knn)
        return x

    def myforward(self, x, adj, adj_knn):
        gamma = self.gamma

        s_i = torch.sigmoid(x @ self.scores[0] + self.bias[0])

        Dk_i = (x @ self.D_k[0] + self.D_bias[0])
        x = (s_i * self.gc1(x, adj) + (1 - s_i) * self.gc1(x, adj_knn)) + (gamma) * Dk_i * self.gc1.w(x)

        x = self.dropout(x)
        embedding = x.clone()

        # output, no relu and dropput here.
        s_o = torch.sigmoid(x @ self.scores[-1] + self.bias[-1])
        Dk_o = (x @ self.D_k[-1] + self.D_bias[-1])
        x = (s_o * self.gc2(x, adj) + (1 - s_o) * self.gc2(x, adj_knn)) + (gamma) * Dk_o * self.gc2.w(x)

        # self.ss = torch.cat((s_i.view(1, -1), s_o.view(1, -1), gamma * Dk_i.view(1, -1), gamma * Dk_o.view(1, -1)), dim=0)
        return x, embedding

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()
        self.w.reset_parameters()

        for s in self.scores:
            stdv = 1. / math.sqrt(s.size(1))
            s.data.uniform_(-stdv, stdv)

        for b in self.bias:
            # fill in b with postive value to make
            # score s closer to 1 at the beginning
            b.data.fill_(0.)

        for Dk in self.D_k:
            stdv = 1. / math.sqrt(Dk.size(1))
            Dk.data.uniform_(-stdv, stdv)

        for b in self.D_bias:
            b.data.fill_(0.)
