import numpy as np
import scipy.sparse as sp
import tensorflow as tf

from numbers import Number
from .shape_utils import repeat, is_iterable

def sample_mask(indices, shape):
    
    if is_iterable(indices): 
        return [sample_mask(index, shape) for index in indices]
    else:
        mask = np.zeros(shape, np.bool)
        mask[indices] = True
        return mask

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).T
        values = mx.data.astype('float32')
        shape = mx.shape
        return coords, values, shape

    if is_iterable(sparse_mx): 
        return [to_tuple(mx) for mx in sparse_mx]
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx


def normalize_adj(adjacency, rate=-0.5, self_loop=True):
    """Normalize adjacency matrix."""
    def normalize(adj, alpha):
        if self_loop:
            adj = adj + sp.eye(adj.shape[0])
        row_sum = adj.sum(1).A1
        d_inv_sqrt = np.power(row_sum, alpha)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return (d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt).astype(np.float32)

    if is_iterable(adjacency):
        size = len(adjacency)
        rate = repeat(rate, size)
        return [normalize(adj_, rate_) for adj_, rate_ in zip(adjacency, rate)]
    else:
        adjacency = normalize(adjacency, rate)

    return adjacency

# ChebyGCN
def chebyshev_polynomials(adj, rate=-0.5, order=3):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""

    assert order >= 2
    adj_normalized = normalize_adj(adj, rate=rate, self_loop=False)
    I = sp.eye(adj.shape[0], dtype=adj.dtype).tocsr()
    laplacian = I - adj_normalized
    largest_eigval = sp.linalg.eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
    scaled_laplacian = (2. / largest_eigval) * laplacian - I

    t_k = []
    t_k.append(I)
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        return 2 * scaled_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, order+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))
        
    return t_k

# FASTGCN
def column_prop(adj):
    column_norm = sp.linalg.norm(adj, axis=0)
    norm_sum = column_norm.sum()
    return column_norm/norm_sum

# def add_self_loop_edge(edge_index, num_nodes, edge_weight=None, fill_weight=1.0):
#     diagnal_edges = [[node_index, node_index] for node_index in range(num_nodes)]
#     diagnal_edge_index = np.array(diagnal_edges).T.astype(np.int32)

#     updated_edge_index = tf.concat([edge_index, diagnal_edge_index], axis=1)

#     if not tf.is_tensor(edge_index):
#         updated_edge_index = updated_edge_index.numpy()

#     if edge_weight is not None:
#         diagnal_edge_weight = tf.cast(tf.fill([num_nodes], fill_weight), tf.float32)
#         updated_edge_weight = tf.concat([edge_weight, diagnal_edge_weight], axis=0)

#         if not tf.is_tensor(edge_weight):
#             updated_edge_weight = updated_edge_weight.numpy()
#     else:
#         updated_edge_weight = None

#     return updated_edge_index, updated_edge_weight


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