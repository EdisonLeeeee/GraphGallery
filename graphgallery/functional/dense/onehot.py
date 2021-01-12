import numpy as np

from ..transforms import Transform
from ..decorators import multiple
from ..get_transform import Transformers

__all__ = ['onehot', 'Onehot']


@Transformers.register()
class Onehot(Transform):

    def __init__(self, depth=None):
        super().__init__()
        self.depth = depth

    def __call__(self, *x):
        return onehot(*x, depth=self.depth)

    def extra_repr(self):
        return f"depth={self.depth}"


@multiple()
def onehot(label, depth=None):
    """Get the one-hot like label of nodes."""
    label = np.asarray(label, dtype=np.int32)
    depth = depth or label.max() + 1
    if label.ndim == 1:
        return np.eye(depth, dtype=label.dtype)[label]
    else:
        raise ValueError(f"label must be a 1D array, but got {label.ndim}D array.")
