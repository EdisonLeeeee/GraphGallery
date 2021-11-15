import torch
import torch.nn as nn
import torch.nn.functional as F

from graphgallery.nn.init import glorot_uniform, zeros


class GATConv(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 attn_heads=8,
                 alpha=0.2,
                 reduction='concat',
                 dropout=0.6,
                 bias=False):
        super().__init__()

        if reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        self.in_features = in_features
        self.out_features = out_features

        self.dropout = dropout
        self.attn_heads = attn_heads
        self.reduction = reduction

        self.kernels = nn.ParameterList()
        self.attn_kernel_self, self.attn_kernel_neighs = nn.ParameterList(
        ), nn.ParameterList()
        self.biases = nn.ParameterList()
        self.bias = bias

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            W = nn.Parameter(torch.FloatTensor(in_features, out_features),
                             requires_grad=True)
            self.kernels.append(W)
            a1 = nn.Parameter(torch.FloatTensor(out_features, 1),
                              requires_grad=True)
            self.attn_kernel_self.append(a1)
            a2 = nn.Parameter(torch.FloatTensor(out_features, 1),
                              requires_grad=True)
            self.attn_kernel_neighs.append(a2)

            if bias:
                self.biases.append(nn.Parameter(torch.Tensor(out_features)))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.reset_parameters()

    def reset_parameters(self):
        for head in range(self.attn_heads):
            W, a1, a2 = self.kernels[head], self.attn_kernel_self[
                head], self.attn_kernel_neighs[head]
            glorot_uniform(W)
            glorot_uniform(a1)
            glorot_uniform(a2)

            if self.bias:
                zeros(self.biases[head])

    def forward(self, x, adj):
        if adj.layout != torch.strided:
            adj = adj.to_dense()

        outputs = []
        for head in range(self.attn_heads):
            W = self.kernels[head]
            a1, a2 = self.attn_kernel_self[head], self.attn_kernel_neighs[head]
            Wh = x.mm(W)
            f_1 = Wh.mm(a1)
            f_2 = Wh.mm(a2)

            edge_e = self.leakyrelu(f_1 + f_2.transpose(0, 1))

            zero_vec = -9e15 * torch.ones_like(edge_e)
            attention = torch.where(adj > 0, edge_e, zero_vec)
            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention,
                                  self.dropout,
                                  training=self.training)
            h_prime = torch.matmul(attention, Wh)

            if self.bias:
                h_prime += self.biases[head]

            outputs.append(h_prime)

        if self.reduction == 'concat':
            output = torch.cat(outputs, dim=1)
        else:
            output = torch.mean(torch.stack(outputs), 0)

        return output

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"


# Sparse Version of `GATConv` layer
# This implementation is much faster than dense GATConv.
class SparseGATConv(nn.Module):
    """
    Sparse version of GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self,
                 in_features,
                 out_features,
                 attn_heads=8,
                 alpha=0.2,
                 reduction='concat',
                 dropout=0.6,
                 bias=False):
        super().__init__()

        if reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        self.in_features = in_features
        self.out_features = out_features

        self.dropout = nn.Dropout(dropout)
        self.attn_heads = attn_heads
        self.reduction = reduction

        self.kernels = nn.ParameterList()
        self.att_kernels = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.bias = bias

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            W = nn.Parameter(torch.Tensor(in_features, out_features))
            self.kernels.append(W)
            a = nn.Parameter(torch.Tensor(2 * out_features, 1))
            self.att_kernels.append(a)

            if bias:
                self.biases.append(nn.Parameter(torch.Tensor(out_features)))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.reset_parameters()

    def reset_parameters(self):
        for head in range(self.attn_heads):
            glorot_uniform(self.kernels[head])
            glorot_uniform(self.att_kernels[head])

            if self.bias:
                zeros(self.biases[head])

    def forward(self, x, adj):

        dv = x.device
        N = x.size()[0]
        edges = adj._indices()

        outputs = []
        for head in range(self.attn_heads):
            W, a = self.kernels[head], self.att_kernels[head]
            Wh = x.mm(W)

            Whij = torch.cat([Wh[edges[0]], Wh[edges[1]]], dim=1)

            edge_e = -self.leakyrelu(torch.matmul(Whij, a))
            attention = self.dropout(self.sparse_softmax(edges, edge_e, N, dv))
            h_prime = self.sparse_matmul(edges, attention, Wh)

            if self.bias:
                h_prime += self.biases[head]

            outputs.append(h_prime)

        if self.reduction == 'concat':
            output = torch.cat(outputs, dim=1)
        else:
            output = torch.mean(torch.stack(outputs), 0)

        return output

    @staticmethod
    def sparse_softmax(edges, edge_e, N, device):
        """Softmax for sparse adjacency matrix"""

        source = edges[0]
        e_max = edge_e.max()
        e_exp = torch.exp(edge_e - e_max)
        e_exp_sum = torch.zeros(N, 1, device=device)
        e_exp_sum.scatter_add_(
            dim=0,
            index=source.unsqueeze(1),
            src=e_exp
        )
        e_exp_sum += 1e-10
        e_softmax = e_exp / e_exp_sum[source]

        return e_softmax

    @staticmethod
    def sparse_matmul(edges, attention, Wh):
        """Matmul for sparse adjacency matrix"""

        source, target = edges
        h_prime = torch.zeros_like(Wh)
        h_prime.scatter_add_(
            dim=0,
            index=source.expand(Wh.size(1), -1).t(),
            src=attention * Wh[target]
        )

        return h_prime

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"


##########################################################################
# The following codes are deprecated since we have a Faster implementation
##########################################################################
# class SpecialSpmmFunction(torch.autograd.Function):
#     """Special function for only sparse region backpropataion layer."""
#     @staticmethod
#     def forward(ctx, indices, values, shape, b):
#         assert indices.requires_grad == False
#         a = torch.sparse.FloatTensor(indices, values, shape)
#         ctx.save_for_backward(a, b)
#         ctx.N = shape[0]
#         return torch.spmm(a, b)

#     @staticmethod
#     def backward(ctx, grad_output):
#         a, b = ctx.saved_tensors
#         grad_values = grad_b = None
#         if ctx.needs_input_grad[1]:
#             grad_a_dense = grad_output.matmul(b.t())
#             edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
#             grad_values = grad_a_dense.view(-1)[edge_idx]
#         if ctx.needs_input_grad[3]:
#             grad_b = a.t().matmul(grad_output)
#         return None, grad_values, None, grad_b


# class SpecialSpmm(nn.Module):
#     def forward(self, indices, values, shape, b):
#         return SpecialSpmmFunction.apply(indices, values, shape, b)


# class SparseGATConv(nn.Module):
#     """
#     Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
#     """

#     def __init__(self,
#                  in_features,
#                  out_features,
#                  attn_heads=8,
#                  alpha=0.2,
#                  reduction='concat',
#                  dropout=0.6,
#                  bias=False):
#         super().__init__()

#         if reduction not in {'concat', 'average'}:
#             raise ValueError('Possbile reduction methods: concat, average')

#         self.in_features = in_features
#         self.out_features = out_features

#         self.dropout = nn.Dropout(dropout)
#         self.attn_heads = attn_heads
#         self.reduction = reduction

#         self.kernels = nn.ParameterList()
#         self.att_kernels = nn.ParameterList()
#         self.biases = nn.ParameterList()
#         self.bias = bias

#         # Initialize weights for each attention head
#         for head in range(self.attn_heads):
#             W = nn.Parameter(torch.Tensor(in_features, out_features))
#             self.kernels.append(W)
#             a = nn.Parameter(torch.Tensor(1, 2 * out_features))
#             self.att_kernels.append(a)

#             if bias:
#                 self.biases.append(nn.Parameter(torch.Tensor(out_features)))

#         self.leakyrelu = nn.LeakyReLU(alpha)
#         self.special_spmm = SpecialSpmm()
#         self.reset_parameters()

#     def reset_parameters(self):
#         for head in range(self.attn_heads):
#             glorot_uniform(self.kernels[head])
#             glorot_uniform(self.att_kernels[head])

#             if self.bias:
#                 zeros(self.biases[head])

#     def forward(self, x, adj):

#         dv = x.device
#         N = x.size()[0]
#         edge = adj._indices()

#         outputs = []
#         for head in range(self.attn_heads):
#             W, a = self.kernels[head], self.att_kernels[head]
#             h = x.mm(W)

#             # Self-attention on the nodes - Shared attention mechanism
#             edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
#             # edge: 2*D x edge_e

#             edge_e = torch.exp(-self.leakyrelu(a.mm(edge_h).squeeze()))

#             e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]),
#                                          torch.ones(size=(N, 1), device=dv))
#             edge_e = self.dropout(edge_e)
#             h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
#             h_prime = h_prime.div(e_rowsum)
#             #             h_prime[torch.isnan(h_prime)] = 0.

#             if self.bias:
#                 h_prime += self.biases[head]

#             outputs.append(h_prime)

#         if self.reduction == 'concat':
#             output = torch.cat(outputs, dim=1)
#         else:
#             output = torch.mean(torch.stack(outputs), 0)

#         return output

#     def __repr__(self):
#         return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"
