import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize, StandardScaler, RobustScaler
from graphgallery.transforms import Transform
from graphgallery.utils.decorators import MultiInputs


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
            graphgallery.transforms.normalize_attr
        """
        return normalize_attr(attr_matrix, norm=self.norm)

    def __repr__(self):
        return f"{self.__class__.__name__}(norm={self.norm})"


@MultiInputs()
def normalize_attr(x, norm='l1'):
    """Normalize the attribute matrix with given type.

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
        A normalized attribute matrix.

    See also
    ----------
        graphgallery.transforms.NormalizeAttr        
    """
    if norm not in {'l1', 'l1_0', 'scale', 'robust_scale', None}:
        raise ValueError(f'{norm} is not a supported norm.')

    if norm == 'l1':
        x_norm = normalize(x, norm='l1', axis=1)
    elif norm == 'l1_0':
        x_norm = normalize(x, norm='l1', axis=0)
    elif norm == 'scale':
        # something goes wrong with type float32
        x_norm = StandardScaler().fit(x).transform(x)
    elif norm == 'robust_scale':
        x_norm = RobustScaler().fit(x).transform(x)
    else:
        x_norm = x.copy()
    return x_norm
