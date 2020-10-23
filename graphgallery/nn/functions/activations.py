import numpy as np
from graphgallery.typing import ArrayLike1D


def softmax(x: ArrayLike1D, axis: int = -1) -> ArrayLike1D:
    """Numpy version of Softmax activation function

    Parameters
    ----------
    x : ArrayLike1D
        Elements to softmax.
    axis : int, optional
        Axis or axes along which a softmax is performed, by default -1.

    Returns
    -------
    softmax_along_axis: ArrayLike1D
        An array with the same shape as `x`.
    """
    exp_x = np.exp(x - np.max(x))
    return exp_x/exp_x.sum(axis=axis, keepdims=True)
