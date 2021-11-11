import collections
import numpy as np

from collections import Iterable
from typing import Any, Optional

__all__ = [
    'is_iterable',
    'is_listlike',
    'is_multiobjects',
    'is_scalar',
    'is_intscalar',
    'is_floatscalar',
]


def is_iterable(x: Any) -> bool:
    """Check whether `x` is an iterable object except for string."""
    return isinstance(x, collections.abc.Iterable) and not isinstance(x, (str, bytes))


def is_listlike(x: Any) -> bool:
    """Check whether `x` is list like, e.g., Tuple, List, or Numpy object.

    Parameters:
    ----------
    x: A python object to check.
    Returns:
    ----------
    `True` iff `x` is a list like sequence.
    """
    return isinstance(x, (list, tuple))


def is_multiobjects(x: Any) -> bool:
    """Check whether `x` is a list of complex objects (not integers).

    Parameters:
    ----------
    x: A python object to check.
    Returns:
    ----------
    `True` iff `x` is a list of complex objects.
    """
    return (is_listlike(x) or (isinstance(x, np.ndarray)
                               and x.dtype == "O")) and len(x) > 0 and not is_scalar(x[0])


def is_scalar(x: Any) -> bool:
    """Check whether `x` is a scalar, an array scalar, or a 0-dim array.
    Parameters:
    ----------
    x: A python object to check.
    Returns:
    ----------
    `True` iff `x` is a scalar, an array scalar, or a 0-dim array.
    """
    return np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0)


def is_intscalar(x: Any) -> bool:
    """Check whether `x` is an integer scalar.
    Parameters:
    ----------
    x: A python object to check.
    Returns:
    ----------
    `True` iff `x` is an integer scalar (built-in or Numpy integer).
    """
    return isinstance(x, (
        int,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ))


def is_floatscalar(x: Any) -> bool:
    """Check whether `x` is a float scalar.

    Parameters:
    ----------
    x: A python object to check.

    Returns:
    ----------
    `True` iff `x` is a float scalar (built-in or Numpy float).
    """
    return isinstance(x, (
        float,
        np.float16,
        np.float32,
        np.float64,
    ))
