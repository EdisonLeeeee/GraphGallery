"""The following codes are borrowed and adapted from torch_geometric

https://github.com/pyg-team/pytorch_geometric
"""


import torch
import random
import numpy as np
from typing import Optional

__all__ = ["negative_sampling", "add_self_loops", "remove_self_loops"]


def sample(high: int, size: int, device=None):
    size = min(high, size)
    return torch.tensor(random.sample(range(high), size), device=device)


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, torch.Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))


def remove_self_loops(edge_index, edge_attr: Optional[torch.Tensor] = None):
    r"""Removes every self-loop in the graph given by :attr:`edge_index`, so
    that :math:`(i,i) \not\in \mathcal{E}` for every :math:`i \in \mathcal{V}`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    if edge_attr is None:
        return edge_index, None
    else:
        return edge_index, edge_attr[mask]


def add_self_loops(edge_index, edge_weight: Optional[torch.Tensor] = None,
                   fill_value: float = 1., num_nodes: Optional[int] = None):
    r"""Adds a self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted, self-loops will be added with edge weights
    denoted by :obj:`fill_value`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_weight (Tensor, optional): One-dimensional edge weights.
            (default: :obj:`None`)
        fill_value (float, optional): If :obj:`edge_weight` is not :obj:`None`,
            will add self-loops with edge weights of :obj:`fill_value` to the
            graph. (default: :obj:`1.`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    N = maybe_num_nodes(edge_index, num_nodes)

    loop_index = torch.arange(0, N, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        loop_weight = edge_weight.new_full((N, ), fill_value)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

    edge_index = torch.cat([edge_index, loop_index], dim=1)

    return edge_index, edge_weight


def negative_sampling(edge_index, num_nodes=None, num_neg_samples=None,
                      method="sparse", force_undirected=False):
    r"""Samples random negative edges of a graph given by :attr:`edge_index`.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        num_neg_samples (int, optional): The (approximate) number of negative
            samples to return. If set to :obj:`None`, will try to return a
            negative edge for every positive edge. (default: :obj:`None`)
        method (string, optional): The method to use for negative sampling,
            *i.e.*, :obj:`"sparse"` or :obj:`"dense"`.
            This is a memory/runtime trade-off.
            :obj:`"sparse"` will work on any graph of any size, while
            :obj:`"dense"` can perform faster true-negative checks.
            (default: :obj:`"sparse"`)
        force_undirected (bool, optional): If set to :obj:`True`, sampled
            negative edges will be undirected. (default: :obj:`False`)

    :rtype: LongTensor
    """
    assert not force_undirected, "currently `force_undirected` argument is not supported."
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    num_neg_samples = num_neg_samples or edge_index.size(1)

    # Do not include self-loops in negative samples
    # TODO: is it necessary?
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    # Handle '|V|^2 - |E| < |E|'.
    size = num_nodes * num_nodes
    num_neg_samples = min(num_neg_samples, size - edge_index.size(1))

    row, col = edge_index

    idx = row * num_nodes + col

    # Percentage of edges to oversample so that we are save to only sample once
    # (in most cases).
    alpha = abs(1 / (1 - 1.1 * (edge_index.size(1) / size)))

    if method == 'dense':
        mask = edge_index.new_ones(size, dtype=torch.bool)
        mask[idx] = False
        mask = mask.view(-1)

        perm = sample(size, int(alpha * num_neg_samples),
                      device=edge_index.device)
        perm = perm[mask[perm]][:num_neg_samples]

    else:
        perm = sample(size, int(alpha * num_neg_samples))
        mask = torch.from_numpy(np.isin(perm, idx.to('cpu'))).to(torch.bool)
        perm = perm[~mask][:num_neg_samples].to(edge_index.device)

    row = torch.div(perm, num_nodes, rounding_mode='floor')
    col = perm % num_nodes
    neg_edge_index = torch.stack([row, col], dim=0).long()

    return neg_edge_index
