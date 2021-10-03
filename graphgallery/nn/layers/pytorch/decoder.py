import torch
import torch.nn as nn
from graphgallery.nn.layers.pytorch import activations


__all__ = ['InnerProductDecoder', 'MergeDecoder']


class InnerProductDecoder(nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder."""

    def __init__(self, activation='sigmoid'):
        super().__init__()
        self.act = activations.get(activation)

    def forward(self, z, edge_index=None):
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.
        """
        if edge_index is None:
            out = torch.matmul(z, z.t())
        else:
            out = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

        return self.act(out)


class MergeDecoder(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features + in_features, in_features)
        self.fc2 = nn.Linear(in_features, out_features)
        self.act = nn.ReLU()

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, z, edge_index):
        x1 = z[edge_index[0]]
        x2 = z[edge_index[1]]
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        return self.fc2(h)
