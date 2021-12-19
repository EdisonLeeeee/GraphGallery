import torch
import torch.nn as nn
from graphgallery.nn.layers.pytorch import activations
from graphgallery.nn.layers.dgl import RobustConv


class RobustGCN(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hids: list = [16],
                 acts: list = ['relu'],
                 dropout: float = 0.5,
                 bias: bool = True,
                 gamma: float = 1.0):
        r"""
        Parameters
        ----------
        in_features : int, 
            the input dimmensions of model
        out_features : int, 
            the output dimensions of model
        hids : list, optional
            the number of hidden units of each hidden layer, by default [16]
        acts : list, optional
            the activaction function of each hidden layer, by default ['relu']
        dropout : float, optional
            the dropout ratio of model, by default 0.5
        bias : bool, optional
            whether to use bias in the layers, by default True
        gamma : float, optional
            the attention weight, by default 1.0
        """

        super().__init__()

        assert len(hids) == len(acts) and len(hids) > 0
        self.conv1 = RobustConv(in_features,
                                hids[0],
                                bias=bias,
                                activation=activations.get(acts[0]))

        conv2 = nn.ModuleList()
        in_features = hids[0]

        for hid, act in zip(hids[1:], acts[1:]):
            conv2.append(RobustConv(in_features,
                                    hid,
                                    bias=bias,
                                    gamma=gamma,
                                    activation=activations.get(act)))
            in_features = hid

        conv2.append(RobustConv(in_features, out_features, gamma=gamma, bias=bias))
        self.conv2 = conv2
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.conv2:
            conv.reset_parameters()

    def forward(self, x, g):
        x = self.dropout(x)
        mean, var = self.conv1(g, x)
        self.mean, self.var = mean, var

        for conv in self.conv2:
            mean, var = self.dropout(mean), self.dropout(var)
            mean, var = conv(g, (mean, var))

        std = torch.sqrt(var + 1e-8)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mean)
        return z
