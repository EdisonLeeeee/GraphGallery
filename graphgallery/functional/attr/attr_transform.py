import numpy as np
import scipy.sparse as sp

from sklearn import preprocessing
from typing import Union

from ..transforms import Transform
from ..decorators import MultiInputs

__all__ = ['augment_attr', 'normalize_attr', 'NormalizeAttr']


def augment_attr(attr_matrix: np.ndarray, N: int,
                 fill_weight: Union[float, list, np.ndarray] = 0.):
    """Augment a specified attribute matrix.

    Examples
    ----------
    >>> augment_attr(attr_matrix, 10, fill_weight=1.0)

    >>> augment_attr(attr_matrix, 10, fill_weight=attr_matrix[-1])

    Parameters
    ----------
    attr_matrix: shape [n_nodes, n_nodes].
        A Scipy sparse adjacency matrix.
    N: number of added nodes.
        node ids [n_nodes, ..., n_nodes+N-1].   
    fill_weight: float or 1D array.
        + float scalar: the weight for the augmented matrix
        + 1D array: repeated N times to augment the matrix.


    """
    if is_scalar(fill_weight):
        M = np.zeros([N, attr_matrix.shape[1]], dtype=attr_matrix.dtype) + fill_weight
    elif isinstance(fill_weight, (list, np.ndarray)):
        fill_weight = fill_weight.astype(attr_matrix.dtype, copy=False)
        M = np.tile(fill_weight, N).reshape([N, -1])
    else:
        raise ValueError(f"Unrecognized input: {fill_weight}.")

    augmented_attr = np.vstack([attr_matrix, M])
    return augmented_attr


class NormalizeAttr(Transform):
    """Normalize the attribute matrix with given type."""

    def __init__(self, norm='l1'):
        """
        Parameters
        ----------
        norm: The specified type for the normalization.
            'l1': l1-norm for axis 1, from `sklearn.preprocessing`.
            'l1_0': l1-norm for axis 0, from `sklearn.preprocessing`.
            'scale': standard scale for axis 0, from 
                `sklearn.preprocessing.scale`
            'robust_scale', robust scale for axis 0, from 
                `sklearn.preprocessing.robust_scale`
            None: return the copy of `x`

        Returns
        ----------
            A normalized attribute matrix.
        """
        super().__init__()
        self.norm = norm

    def __call__(self, attr_matrix):
        """
        Parameters
        ----------
        attr_matrix: [N, M], Numpy array-like matrix

        Returns
        ----------
        A Normalized attribute matrix.

        See also
        ----------
        graphgallery.functional.normalize_attr
        """
        return normalize_attr(attr_matrix, norm=self.norm)

    def __repr__(self):
        """
        Return a repr representation of a repr__.

        Args:
            self: (todo): write your description
        """
        return f"{self.__class__.__name__}(norm={self.norm})"


@MultiInputs()
def normalize_attr(x, norm='l1'):
    """Normalize a matrix with given type.

    Parameters
    ----------
    x: Numpy array-like matrix
    norm: The specified type for the normalization.
        'l1': l1-norm for axis 1, from `sklearn.preprocessing`.
        'l1_0': l1-norm for axis 0, from `sklearn.preprocessing`.
        'scale': standard scale for axis 0, from 
            `sklearn.preprocessing.scale`
        'robust_scale', robust scale for axis 0, from 
            `sklearn.preprocessing.robust_scale`
        None: return the copy of `x`

    Returns
    ----------
    A normalized matrix.
    """
    if norm not in {'l1', 'l1_0', 'scale', 'robust_scale', None}:
        raise ValueError(f'{norm} is not a supported norm.')

    if norm == 'l1':
        x_norm = preprocessing.normalize(x, norm='l1', axis=1)
    elif norm == 'l1_0':
        x_norm = preprocessing.normalize(x, norm='l1', axis=0)
    elif norm == 'scale':
        # something goes wrong with type float32
        x_norm = preprocessing.StandardScaler().fit(x).transform(x)
    elif norm == 'robust_scale':
        x_norm = preprocessing.RobustScaler().fit(x).transform(x)
    else:
        x_norm = x.copy()
    return x_norm
