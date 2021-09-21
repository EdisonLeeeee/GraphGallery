import warnings
import numpy as np
from ..edge_level import asedge

__all__ = ["flip_attr"]


def flip_attr(matrix, flips):
    if flips is None or len(flips) == 0:
        warnings.warn(
            "There is NO flips, the matrix stays the same.",
            UserWarning,
        )
        return matrix.copy()

    matrix = matrix.copy()
    flips = asedge(flips, shape="row_wise").T
    row, col = flips
    matrix[matrix < 0] = 0.
    matrix[matrix > 1] = 1.
    matrix[row, col] = 1. - matrix[row, col]
    return matrix
