import torch
from graphgallery.nn.layers.pytorch import activations


class InnerProductDecoder(torch.nn.Module):
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
