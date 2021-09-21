import numpy as np

from ..base_transforms import DenseTransform
from ..decorators import multiple
from ..transform import Transform

__all__ = ['onehot', 'Onehot']


@Transform.register()
class Onehot(DenseTransform):

    def __init__(self, depth=None):
        super().__init__()
        self.collect(locals())

    def __call__(self, *x):
        return onehot(*x, depth=self.depth)


@multiple()
def onehot(label, depth=None):
    """Get the one-hot like label of nodes."""
    label = np.asarray(label, dtype=np.int32)
    depth = depth or label.max() + 1
    if label.ndim == 1:
        return np.eye(depth, dtype=label.dtype)[label]
    else:
        raise ValueError(f"label must be a 1D array, but got {label.ndim}D array.")
