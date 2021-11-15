import numpy as np
from ..transform import DenseTransform
from ..decorators import multiple
from ..transform import Transform

__all__ = ["Softmax", "softmax"]


@Transform.register()
class Softmax(DenseTransform):
    """Numpy version of Softmax activation function"""

    def __init__(self, axis: int = -1):
        """
        Parameters
        ----------
        axis : int, optional
            Axis or axes along which a softmax is performed, by default -1.


        """
        super().__init__()
        self.collect(locals())

    def __call__(self, *x):
        """
        Parameters
        ----------
        x : np.ndarray
            Elements to softmax.
        """
        return softmax(*x, axis=self.axis)


@multiple()
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
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
    return exp_x / exp_x.sum(axis=axis, keepdims=True)
