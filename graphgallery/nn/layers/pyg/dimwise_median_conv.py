import torch
import torch.nn as nn
import dgl.function as fn
from dgl import DGLError
from graphwar.utils.normalize import dgl_normalize
try:
    from glcore import dimmedian_idx
except ModuleNotFoundError:
    dimmedian_idx = None
    
class DimwiseMedianConv(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 add_self_loop=True,
                 norm='none',
                 activation=None,
                 weight=True,
                 bias=True):

        super().__init__()
        if norm not in ('none', 'both', 'right', 'left'):
            raise DGLError('Invalid norm value. Must be either "none", "both", "right" or "left".'
                           ' But got "{}".'.format(norm))
            
        if dimmedian_idx is None:
            raise RuntimeWarning("Module 'glcore' is not properly installed, please refer to "
                     "'https://github.com/EdisonLeeeee/glcore' for more information.")
            
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._add_self_loop = add_self_loop
        self._activation = activation

        if weight:
            self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

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
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, graph, feat, edge_weight=None):
        r"""

        Description
        -----------
        Compute Graph Convolution layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.
        edge_weight : torch.Tensor, optional
            Optional edge weight for each edge.            

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """

        assert edge_weight is None or edge_weight.size(0) == graph.num_edges()

        if self._add_self_loop:
            graph = graph.add_self_loop()
            if edge_weight is not None:
                size = (graph.num_nodes(),) + edge_weight.size()[1:]
                self_loop = edge_weight.new_ones(size)
                edge_weight = torch.cat([edge_weight, self_loop])
        else:
            graph = graph.local_var()

        edge_weight = dgl_normalize(graph, self._norm, edge_weight)
        if self.weight is not None:
            feat = feat @ self.weight        
        
        #========= weighted dimension-wise Median aggregation ===
        N, D = feat.size()
        edge_index = torch.stack(graph.edges(), dim=0)
        median_idx = dimmedian_idx(feat, edge_index, edge_weight, N)
        col_idx = torch.arange(D, device=graph.device).view(1, -1).expand(N, D)
        feat = feat[median_idx, col_idx]
        print(edge_index.size())
        print(median_idx.size())
        #========================================================
        torch.cuda.empty_cache()
        
        if self.bias is not None:
            feat = feat + self.bias

        if self._activation is not None:
            feat = self._activation(feat)
        return feat

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)