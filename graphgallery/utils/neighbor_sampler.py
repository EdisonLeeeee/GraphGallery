import torch
import scipy.sparse as sp
from graphgallery.sampler import neighbor_sampler_cpu


class NeighborSampler:
    """Neighbor sampler as in `GraphSAGE`:
    `Inductive Representation Learning on Large Graphs <https://arxiv.org/abs/1706.02216>`

    Parameters:
    -----------
    adj_matrix：scipy.spars.csr_matrix, the input matrix to be sampled

    Examples
    --------
    >>> from graphgallery.utils import NeighborSampler
    >>> sampler = NeighborSampler(adj)
    >>> sampler.sample(torch.arange(100), size=3)


    Note:
    -----
    Please make sure there is not dangling nodes, otherwise there would be an error.
    """

    def __init__(self, adj_matrix: sp.csr_matrix):
        self.rowptr = torch.LongTensor(adj_matrix.indptr)
        self.col = torch.LongTensor(adj_matrix.indices)
        self.data = torch.FloatTensor(adj_matrix.data)

    def sample(self, nodes, size, *, return_weight=False, as_numpy=False, replace=True):
        """Sample local neighborhood from input batch nodes

        Parameters:
        -----------
        nodes: torch.LongTensor or numpy.array, the input root nodes
        size: int, the number of neighbors sampled for each node, `-1` means the whole neighbor set
        return_weight: bool, if True, return the sampled edges weights for each pair (node, neighbor)
        as_numpy: bool, if True, return numpy array, otherwise return torch.tensor
        replace: bool, whether the sample is with or without replacement

        returns:
        --------
        (targets, neighbors) if return_weight=False
        (targets, neighbors, edge_weights) if return_weight=True

        Note:
        -----
        The outputs would be `torch.tensor` by default, 
        if you want to return numpy array, set `as_numpy=True`.

        """
        if not torch.is_tensor(nodes):
            nodes = torch.LongTensor(nodes)

        targets, neighbors, e_id = neighbor_sampler_cpu(self.rowptr, self.col, nodes, size, replace)

        if return_weight:
            outputs = targets, neighbors, self.data[e_id]
        else:
            outputs = targets, neighbors

        if as_numpy:
            outputs = tuple(out.numpy() for out in outputs)

        return outputs
