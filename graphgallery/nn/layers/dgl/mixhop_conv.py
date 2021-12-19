import torch
import torch.nn as nn
import dgl.function as fn


class MixHopConv(nn.Module):
    r"""

    Description
    -----------
    MixHop Graph Convolutional layer from paper `MixHop: Higher-Order Graph Convolutional Architecturesvia Sparsified Neighborhood Mixing
     <https://arxiv.org/abs/1905.00067>`__.

    .. math::
        H^{(i+1)} =\underset{j \in P}{\Bigg\Vert} \sigma\left(\widehat{A}^j H^{(i)} W_j^{(i)}\right),

    where :math:`\widehat{A}` denotes the symmetrically normalized adjacencymatrix with self-connections,
    :math:`D_{ii} = \sum_{j=0} \widehat{A}_{ij}` its diagonal degree matrix,
    :math:`W_j^{(i)}` denotes the trainable weight matrix of different MixHop layers.

    Parameters
    ----------
    in_features : int
        Input feature size. i.e, the number of dimensions of :math:`H^{(i)}`.
    out_features : int
        Output feature size for each power.
    bias: bool, optional
        bias term for each layer
    p: list, optional
        List of powers of adjacency matrix. Defaults: ``[0, 1, 2]``.
    activation: callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    """

    def __init__(self,
                 in_features,
                 out_features,
                 bias=False,
                 p=[0, 1, 2],
                 activation=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.p = p
        self.activation = activation

        # define weight dict for each power j
        self.weights = nn.ModuleDict({
            str(j): nn.Linear(in_features, out_features, bias=bias) for j in p
        })

    def forward(self, graph, feats):
        with graph.local_scope():
            # assume that the graphs are undirected and graph.in_degrees() is the same as graph.out_degrees()
            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5).to(feats.device).unsqueeze(1)
            max_j = max(self.p) + 1
            outputs = []
            for j in range(max_j):

                if j in self.p:
                    output = self.weights[str(j)](feats)
                    outputs.append(output)

                feats = feats * norm
                graph.ndata['h'] = feats
                graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                feats = graph.ndata.pop('h')
                feats = feats * norm

            final = torch.cat(outputs, dim=1)

            if self.activation is not None:
                final = self.activation(final)

            return final

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features}, p={self.p})"
