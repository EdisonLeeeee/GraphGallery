import torch
import math
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn import Module, ModuleList, LeakyReLU
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
        self.concat = attn_heads_reduction
        self.activation = get_activation(activation)

        self.kernels = ModuleList()
        self.attn_kernels = ModuleList()

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            W = Parameter(torch.empty(size=(in_channels, out_channels)))
            self.kernels.append(W)
            a = Parameter(torch.empty(size=(2 * out_channels, 1)))
            self.attn_kernels.append(a)

        self.leakyrelu = LeakyReLU(self.alpha)

        # ????? if use_bias:
        self.reset_parameters()

    def reset_parameters(self):
        for head in range(self.attn_heads):
            W, a = kernels[head], attn_kernels[head]
            xavier_uniform(self.W.data, gain=1.414)
            xavier_uniform(self.a.data, gain=1.414)

        # zeros(self.bias)

    def forward(self, inputs):
        x, adj = inputs

        outputs = []
        for head in range(self.attn_heads):
            W, a = kernels[head], attn_kernels[head]
            Wh = torch.mm(x, W)
            a_input = self._prepare_attention_mechanism_input(Wh)
            e = self.leakyrelu(torch.matmul(a_input, a).squeeze(2))

            zero_vec = -9e15*torch.ones_like(e)
            attention = torch.where(adj > 0, zero_vec)
            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention, self.dropout, training=self.training)
            h_prime = torch.matmul(attention, Wh) # spmm?

            # ??? TF那里好像没有这一步？
            # if self.attn_heads_reduction == 'concat':
            #     outputs.append(F.elu(h_prime))
            # else:
            outputs.append(h_prime)

        if self.attn_heads_reduction == 'concat':
            output = torch.cat(outputs, dim=1)

        return self.activation(output)

    # copied
    def _prepare_attentional_mechanism_input(self, Wh):
            N = Wh.size()[0] # number of nodes
    
            Wh_repeated_in_chunks = Wh.repeat(1, N).view(N * N, self.out_channels)
            Wh_repeated_alternating = Wh.repeat(N, 1)
    
            all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
    
            return all_combinations_matrix.view(N, N, 2 * self.out_channels)
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_channels) + ' -> ' \
            + str(self.out_channels) + ')'
