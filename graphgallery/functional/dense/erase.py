import numpy as np
from ..functions import index_to_mask

__all__ = ["erase_node_attr", "erase_node_attr_except"]


def erase_node_attr(node_attr, nodes, missing_rate=0.1):
    """PairNorm: Tackling Oversmoothing in GNNs
    <https://openreview.net/forum?id=rkecl1rtwB>
    ICLR 2020"""
    assert 0 <= missing_rate <= 1
    if missing_rate > 0:
        node_erased = np.random.choice(nodes, size=int(len(nodes) * missing_rate), replace=False)
        node_attr = node_attr.copy()
        node_attr[node_erased] = 0.
    return node_attr


def erase_node_attr_except(node_attr, nodes, missing_rate=0.1):
    """PairNorm: Tackling Oversmoothing in GNNs
    <https://openreview.net/forum?id=rkecl1rtwB>
    ICLR 2020"""
    num_nodes = node_attr.shape[0]
    mask = index_to_mask(nodes, num_nodes)
    erased = np.arange(num_nodes, dtype=nodes.dtype)
    return erase_node_attr(node_attr, erased[~mask], missing_rate=missing_rate)
