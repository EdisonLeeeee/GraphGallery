import numpy as np


def softmax(x: np.ndarray, axis=-1)->np.ndarray:
    """Numpy version of Softmax activation function

    Parameters
    ----------
    x : np.ndarray
        Elements to softmax.
    axis : int, optional
        Axis or axes along which a softmax is performed, by default -1.

    Returns
    -------
    softmax_along_axis: np.ndarray
        An array with the same shape as `x`.
    """
    exp_x = np.exp(x - np.max(x))
    return exp_x/exp_x.sum(axis=axis, keepdims=True)
