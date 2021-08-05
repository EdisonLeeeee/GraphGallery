import numpy as np
from typing import Optional

__all__ = ['maybe_shape', 'maybe_num_nodes']


def maybe_shape(edge):
    assert np.ndim(edge) == 2 and np.shape(edge)[0] == 2
    M = np.max(edge[0]) + 1
    N = np.max(edge[1]) + 1
    M = N = np.maximum(M, N)
    return M, N


def maybe_num_nodes(index: np.ndarray,
                    num_nodes: Optional[int] = None) -> int:

    return int(np.max(index)) + 1 if num_nodes is None else num_nodes
