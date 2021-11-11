import numpy as np
from ..functions import index_to_mask

__all__ = ["erase_node_attr", "erase_node_attr_except"]


def erase_node_attr(attr_matrix, nodes, missing_rate=0.1):
    """PairNorm: Tackling Oversmoothing in GNNs
    <https://openreview.net/forum?id=rkecl1rtwB>
    ICLR 2020"""
    assert 0 <= missing_rate <= 1
    if missing_rate > 0:
        node_erased = np.random.choice(nodes, size=int(len(nodes) * missing_rate), replace=False)
        attr_matrix = attr_matrix.copy()
        attr_matrix[node_erased] = 0.
    return attr_matrix


def erase_node_attr_except(attr_matrix, nodes, missing_rate=0.1):
    """PairNorm: Tackling Oversmoothing in GNNs
    <https://openreview.net/forum?id=rkecl1rtwB>
    ICLR 2020"""
    num_nodes = attr_matrix.shape[0]
    mask = index_to_mask(nodes, num_nodes)
    erased = np.arange(num_nodes, dtype=nodes.dtype)
    return erase_node_attr(attr_matrix, erased[~mask], missing_rate=missing_rate)
