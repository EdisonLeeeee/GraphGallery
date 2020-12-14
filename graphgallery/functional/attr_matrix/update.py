import warnings
import numpy as np
from ..edge_level import asedge

__all__ = ["flip_attr"]


def flip_attr(matrix, flips):
    if flips is None or len(flips) == 0:
        warnings.warn(
            "There is NO flips, the matrix stays the same.",
            RuntimeWarning,
        )
        return matrix.copy()

    matrix = matrix.copy()
    flips = asedge(flips)
    matrix[matrix < 0] = 0.
    matrix[matrix > 1] = 1.
    matrix[flips] = 1. - matrix[flips]
    return matrix
