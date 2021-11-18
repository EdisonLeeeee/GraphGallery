import numpy as np
import scipy.sparse as sp

from sklearn import preprocessing
from typing import Union

import graphgallery as gg
from ..transform import DenseTransform
from ..decorators import multiple
from ..transform import Transform

__all__ = ['augment_feat', 'normalize_feat', 'NormalizeFeat']


def augment_feat(attr_matrix: np.ndarray,
                 N: int,
                 fill_weight: Union[float, list, np.ndarray] = 0.):
    """Augment a specified node feature matrix.

    Examples
    ----------
    >>> augment_feat(attr_matrix, 10, fill_weight=1.0)

    >>> augment_feat(attr_matrix, 10, fill_weight=attr_matrix[-1])

    Parameters
    ----------
    attr_matrix: shape [num_nodes, num_nodes].
        A Scipy sparse adjacency matrix.
    N: number of added nodes.
        node ids [num_nodes, ..., num_nodes+N-1].   
    fill_weight: float or 1D array.
        + float scalar: the weight for the augmented matrix
        + 1D array: repeated N times to augment the matrix.


    """
    if gg.is_scalar(fill_weight):
        M = np.zeros([N, attr_matrix.shape[1]],
                     dtype=attr_matrix.dtype) + fill_weight
    elif isinstance(fill_weight, (list, np.ndarray)):
        fill_weight = fill_weight.astype(attr_matrix.dtype, copy=False)
        M = np.tile(fill_weight, N).reshape([N, -1])
    else:
        raise ValueError(f"Unrecognized input: {fill_weight}.")

    augmented_feat = np.vstack([attr_matrix, M])
    return augmented_feat


@Transform.register()
class NormalizeFeat(DenseTransform):
    """Normalize the node feature matrix with given type."""

    def __init__(self, norm='l1'):
        """
        Parameters
        ----------
        norm: The specified type for the normalization.
            'l1': l1-norm for axis 1, from `sklearn.preprocessing`.
            'l1_0': l1-norm for axis 0, from `sklearn.preprocessing`.
            'zscore': standard scale for axis 0, from 
                `sklearn.preprocessing.scale`
            'robust_scale', robust scale for axis 0, from 
                `sklearn.preprocessing.robust_scale`
            None: return the copy of `x`

        Returns
        -------
            A normalized node feature matrix.
        """
        super().__init__()
        self.collect(locals())

    def __call__(self, *x):
        """
        Parameters
        ----------
        x: [N, M], Numpy array-like matrix

        Returns
        -------
        A Normalized feature matrix.

        See also
        --------
        graphgallery.functional.normalize_feat
        """
        return normalize_feat(*x, norm=self.norm)


@multiple()
def normalize_feat(x, *, norm='l1'):
    """Normalize feature matrix with given type.

    Parameters
    ----------
    x: Numpy array-like matrix
    norm: The specified type for the normalization.
        'l1': l1-norm for axis 1, from `sklearn.preprocessing`.
        'l1_0': l1-norm for axis 0, from `sklearn.preprocessing`.
        'zscore': standard scale for axis 0, 
            from `sklearn.preprocessing.scale`
        'robust_scale', robust scale for axis 0, 
            from `sklearn.preprocessing.robust_scale`
        None: return the copy of `x`
        or a callable function

    Returns
    -------
    A normalized feature matrix in Numpy format.
    """
    if callable(norm):
        return norm(x)
    if norm not in {'l1', 'l1_0', 'zscore', 'robust_scale', None}:
        raise ValueError(f'{norm} is not a supported norm.')

    if norm == 'l1':
        x_norm = preprocessing.normalize(x, norm='l1', axis=1)
    elif norm == 'l1_0':
        x_norm = preprocessing.normalize(x, norm='l1', axis=0)
    elif norm == 'zscore':
        # something goes wrong with type float32
        x_norm = preprocessing.StandardScaler().fit(x).transform(x)
    elif norm == 'robust_scale':
        x_norm = preprocessing.RobustScaler().fit(x).transform(x)
    else:
        x_norm = x.copy()
    return x_norm
