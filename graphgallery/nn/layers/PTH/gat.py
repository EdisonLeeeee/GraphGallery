import torch
import math
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn import Module, ParameterList, LeakyReLU
import torch.nn.functional as F

from graphgallery.nn.models.get_activation import get_activation
from graphgallery.nn.init import xavier_uniform, zeros

class GraphAttention(Module):
    def __init__(self, in_channels, out_channels, attn_heads=1, attn_heads_reduction='concat', dropout=0.5, activation=None, use_bias=False):
        super().__init__()

        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.dropout = dropout
        self.alpha = 0.2
        self.attn_heads = attn_heads
        self.attn_heads_reduction = attn_heads_reduction
        self.activation = get_activation(activation)

        self.kernels = ParameterList()
        self.attn_kernel_self, self.attn_kernel_neighs = ParameterList(), ParameterList()
        self.biases = ParameterList()
        self.use_bias = use_bias

        if not use_bias:
            self.register_parameter('bias', None)


        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            W = Parameter(torch.FloatTensor(in_channels, out_channels), requires_grad=True)
            self.kernels.append(W)
            a1 = Parameter(torch.FloatTensor(out_channels, 1), requires_grad=True)
            self.attn_kernel_self.append(a1)
            a2 = Parameter(torch.FloatTensor(out_channels, 1), requires_grad=True)
            self.attn_kernel_neighs.append(a2)

            if use_bias:
                bias = Parameter(torch.Tensor(out_channels))
                self.biases.append(bias)

        self.leakyrelu = LeakyReLU(self.alpha)

        self.reset_parameters()

    def reset_parameters(self):
        for head in range(self.attn_heads):
            W, a1, a2 = self.kernels[head], self.attn_kernel_self[head], self.attn_kernel_neighs[head]
            xavier_uniform(W.data, gain=1.414)
            xavier_uniform(a1.data, gain=1.414)
            xavier_uniform(a2.data, gain=1.414)

            if self.use_bias:
                zeros(self.biases[head])

    def forward(self, inputs):
        x, adj = inputs

        outputs = []
        for head in range(self.attn_heads):
            W, a1, a2 = self.kernels[head], self.attn_kernel_self[head], self.attn_kernel_neighs[head]
            Wh = torch.mm(x, W)

            f_1 = Wh @ a1
            f_2 = Wh @ a2

            e = self.leakyrelu(f_1 + f_2.transpose(0, 1))

            zero_vec = -9e15*torch.ones_like(e)
            attention = torch.where(adj.to_dense() > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention, self.dropout, training=self.training)
            h_prime = torch.matmul(attention, Wh) 

            if self.use_bias:
                h_prime += self.biases[head]

            outputs.append(h_prime)

        if self.attn_heads_reduction == 'concat':
            output = torch.cat(outputs, dim=1)
        else:
            output = torch.mean(torch.stack(outputs), 0)

        return self.activation(output)
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_channels) + ' -> ' \
            + str(self.out_channels) + ')'
