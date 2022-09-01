import torch
import scipy.sparse as sp
from torch import Tensor
from typing import List, Optional, Tuple, NamedTuple, Union

try:
    from glcore import neighbor_sampler_cpu
except (ModuleNotFoundError, ImportError):
    neighbor_sampler_cpu = None


class NeighborSampler:
    """Neighbor sampler as in `GraphSAGE`:
    `Inductive Representation Learning on Large Graphs <https://arxiv.org/abs/1706.02216>`

    Parameters:
    -----------
    adj_matrix：scipy.spars.csr_matrix, the input matrix to be sampled

    Examples
    --------
    >>> from graphgallery.data import NeighborSampler
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

        if neighbor_sampler_cpu is None:
            raise RuntimeWarning("'neighbor_sampler_cpu' is not installed, please refer to "
                                 "'https://github.com/EdisonLeeeee/glcore' for more information.")

        targets, neighbors, e_id = neighbor_sampler_cpu(self.rowptr, self.col, nodes, size, replace)

        if return_weight:
            outputs = targets, neighbors, self.data[e_id]
        else:
            outputs = targets, neighbors

        if as_numpy:
            outputs = tuple(out.numpy() for out in outputs)

        return outputs


class EdgeIndex(NamedTuple):
    edge_index: Tensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        edge_index = self.edge_index.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return EdgeIndex(edge_index, e_id, self.size)


class PyGNeighborSampler:
    def __init__(self, edge_index: Tensor,
                 num_nodes: Optional[int] = None):

        edge_index = edge_index.to('cpu')

        # Obtain a *transposed* `SparseTensor` instance.
        num_nodes = int(edge_index.max()) + 1
        from torch_sparse import SparseTensor
        self.adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                                  value=None,
                                  sparse_sizes=(num_nodes, num_nodes)).t()

        self.adj_t.storage.rowptr()

    def sample(self, batch, sizes: List[int] = [-1]):
        batch_size: int = len(batch)

        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)

        adjs = []
        n_id = batch
        for size in sizes:
            adj_t, n_id = self.adj_t.sample_adj(n_id, size, replace=False)
            e_id = adj_t.storage.value()
            size = adj_t.sparse_sizes()[::-1]

            row, col, _ = adj_t.coo()
            edge_index = torch.stack([col, row], dim=0)
            adjs.append(EdgeIndex(edge_index, e_id, size))

        adjs = adjs[0] if len(adjs) == 1 else adjs[::-1]
        out = (batch_size, n_id, adjs)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}'
