import numpy as np
import scipy.sparse as sp
import tensorflow as tf

from numbers import Number
from sklearn.preprocessing import scale, normalize, robust_scale
from functools import partial

from graphgallery.utils.shape_utils import repeat
from graphgallery.utils.type_check import is_list_like
from graphgallery import config


def sample_mask(indices, shape):
    mask = np.zeros(shape, np.bool)
    mask[indices] = True
    return mask

def normalize_fn(norm_type='row_wise'):
    """Return a normalize function applied for feature matrix
    
    Arguments
    ----------

    norm_type: The specified type for the normalization.
        `row_wise`: l1-norm for axis 1, from sklearn.preprocessing.normalize
        `col_wise`: l1-norm for axis 0, sklearn.preprocessing.normalize
        `scale`: standard scale for axis 0, sklearn.preprocessing.scale
        `robust_scale`, robust scale for axis 0, sklearn.preprocessing.robust_scale
        None: return None

    Returns
    ----------

        A normalize function applied for feature matrix
    """
    assert norm_type in {'row_wise', 'col_wise', 'scale', 'robust_scale', None}

    if norm_type == 'row_wise':
        norm_fn = partial(normalize, axis=1, norm='l1')
    elif norm_type == 'col_wise':
        norm_fn = partial(normalize, axis=0, norm='l1')
    elif norm_type == 'scale':
        norm_fn = lambda x: scale(x.astype('float64')).astype(config.floatx())
    elif norm_type == 'robust_scale':
        norm_fn =  robust_scale        
    else:
        norm_fn = None
    return norm_fn


def normalize_adj(adjacency, rate=-0.5, add_self_loop=True):
    """Normalize adjacency matrix."""
    def normalize(adj, alpha):
        
        if add_self_loop:
            adj = adj + sp.eye(adj.shape[0])
            
        if alpha is None:
            return adj.astype(config.floatx(), copy=False)            

        row_sum = adj.sum(1).A1
        d_inv_sqrt = np.power(row_sum, alpha)
#         d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return (d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt).astype(config.floatx(), copy=False)

    if is_list_like(adjacency) and not isinstance(adjacency[0], Number):
        size = len(adjacency)
        rate = repeat(rate, size)
        return [normalize(adj_, rate_) for adj_, rate_ in zip(adjacency, rate)]
    else:
        adjacency = normalize(adjacency, rate)

    return adjacency

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
        super().__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        # Bunch pickles generated with scikit-learn 0.16.* have an non
        # empty __dict__. This causes a surprising behaviour when
        # loading these pickles scikit-learn 0.17: reading bunch.key
        # uses __dict__ but assigning to bunch.key use __setattr__ and
        # only changes bunch['key']. More details can be found at:
        # https://github.com/scikit-learn/scikit-learn/issues/6196.
        # Overriding __setstate__ to be a noop has the effect of
        # ignoring the pickled __dict__
        pass

