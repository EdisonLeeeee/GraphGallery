import torch
import itertools

import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from typing import Any, Optional
from numbers import Number

import graphgallery as gg

__all__ = ['asintarr', 'indices2mask', 'repeat', 'get_length', 'Bunch']


def asintarr(x: Any, dtype: Optional[str] = None) -> np.ndarray:
    """Convert `x` to interger Numpy array.

    Parameters:
    ----------
    x: Tensor, Scipy sparse matrix,
        Numpy array-like, etc.

    Returns:
    ----------
    Integer Numpy array with dtype or `graphgallery.intx()`

    """
    if dtype is None:
        dtype = gg.intx()
        
    if gg.is_tensor(x, backend="tensorflow"):
        if x.dtype != dtype:
            return tf.cast(x, dtype=dtype)
        else:
            return x
        
    if gg.is_tensor(x, backend="torch"):
        if x.dtype != dtype:
            return x.to(getattr(torch, dtype))
        else:
            return x       

    if gg.is_intscalar(x):
        x = np.asarray([x], dtype=dtype)
    elif gg.is_listlike(x) or isinstance(x, (np.ndarray, np.matrix)):
        x = np.asarray(x, dtype=dtype)
    else:
        raise ValueError(
            f"Invalid input which should be either array-like or integer scalar, but got {type(x)}.")
    return x


def indices2mask(indices: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Convert an array of indices to a boolean array.

    Args:
        indices: (array): write your description
        np: (todo): write your description
        ndarray: (array): write your description
        shape: (int): write your description
    """
    mask = np.zeros(shape, dtype=gg.boolx())
    mask[indices] = True
    return mask

def repeat(src: Any, length: int) -> Any:
    """
    Returns a list of the given length.

    Args:
        src: (todo): write your description
        length: (int): write your description
    """
    if src is None:
        return [None for _ in range(length)]
    if src == [] or src == ():
        return []
    if isinstance(src, (Number, str)):
        return list(itertools.repeat(src, length))
    if (len(src) > length):
        return src[:length]
    if (len(src) < length):
        return list(src) + list(itertools.repeat(src[-1], length - len(src)))
    return src


def get_length(obj: Any) -> int:
    """
    Return length of the length.

    Args:
        obj: (todo): write your description
    """
    if gg.is_iterable(obj):
        length = len(obj)
    else:
        length = 1
    return length

class Bunch(dict):
    """Container object for datasets
    Dictionary-like object that exposes its keys as attributes.
    >>> b = Bunch(a=1, b=2)
    >>> b['b']
    2
    >>> b.b
    2
    >>> b.a = 3
    >>> b['a']
    3
    >>> b.c = 6
    >>> b['c']
    6
    """

    def __init__(self, **kwargs):
        """
        Initialize the class.

        Args:
            self: (todo): write your description
        """
        super().__init__(kwargs)

    def __setattr__(self, key, value):
        """
        Sets the value of a key.

        Args:
            self: (todo): write your description
            key: (str): write your description
            value: (todo): write your description
        """
        self[key] = value

    def __dir__(self):
        """
        Return a list of directories.

        Args:
            self: (todo): write your description
        """
        return self.keys()

    def __getattr__(self, key):
        """
        Returns the value of the given attribute.

        Args:
            self: (todo): write your description
            key: (str): write your description
        """
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        """
        Sets the state should be a boolean.

        Args:
            self: (todo): write your description
            state: (dict): write your description
        """
        # Bunch pickles generated with scikit-learn 0.16.* have an non
        # empty __dict__. This causes a surprising behaviour when
        # loading these pickles scikit-learn 0.17: reading bunch.key
        # uses __dict__ but assigning to bunch.key use __setattr__ and
        # only changes bunch['key']. More details can be found at:
        # https://github.com/scikit-learn/scikit-learn/issues/6196.
        # Overriding __setstate__ to be a noop has the effect of
        # ignoring the pickled __dict__
        pass
