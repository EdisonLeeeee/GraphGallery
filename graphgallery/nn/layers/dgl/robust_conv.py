import torch.nn as nn
import torch.nn.functional as F
import torch as th
import dgl.function as fn
from torch.nn import init


class RobustConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 bias=False, gamma=1.0,
                 activation=None):
        super().__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.weight_mean = nn.Parameter(th.Tensor(in_feats, out_feats))
        self.weight_var = nn.Parameter(th.Tensor(in_feats, out_feats))

        if bias:
            self.bias_mean = nn.Parameter(th.Tensor(out_feats))
            self.bias_var = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias_mean', None)
            self.register_parameter('bias_var', None)

        self._gamma = gamma
        self._activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The model parameters are initialized as in the
        `original implementation <https://github.com/tkipf/gcn/blob/master/gcn/layers.py>`__
        where the weight :math:`W^{(l)}` is initialized using Glorot uniform initialization
        and the bias is initialized to be zero.

        """
        init.xavier_uniform_(self.weight_mean)
        init.xavier_uniform_(self.weight_var)
        if self.bias_mean is not None:
            init.zeros_(self.bias_mean)
            init.zeros_(self.bias_var)

    def forward(self, graph, feat):
        if not isinstance(feat, tuple):
            feat = (feat, feat)

        mean = th.matmul(feat[0], self.weight_mean)
        var = th.matmul(feat[1], self.weight_var)

        if self.bias_mean is not None:
            mean = mean + self.bias_mean
            var = var + self.bias_var

        mean = F.relu(mean)
        var = F.relu(var)

        attention = th.exp(-self._gamma * var)

        degs = graph.in_degrees().float().clamp(min=1)
        norm1 = th.pow(degs, -0.5).to(mean.device).unsqueeze(1)
        norm2 = norm1.square()

        with graph.local_scope():
            graph.ndata['mean'] = mean * attention * norm1
            graph.ndata['var'] = var * attention * attention * norm2
            graph.update_all(fn.copy_src('mean', 'm'), fn.sum('m', 'mean'))
            graph.update_all(fn.copy_src('var', 'm'), fn.sum('m', 'var'))

            mean = graph.ndata.pop('mean') * norm1
            var = graph.ndata.pop('var') * norm2

            if self._activation is not None:
                mean = self._activation(mean)
                var = self._activation(var)

        return mean, var
