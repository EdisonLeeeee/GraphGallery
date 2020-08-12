try:
    import texttable
except ImportError:
    texttable = None
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from graphgallery.utils.data_utils import normalize_adj

# ChebyGCN
def chebyshev_polynomials(adj, rate=-0.5, order=3):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""

    assert order >= 2
    adj_normalized = normalize_adj(adj, rate=rate, self_loop=1.0)
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

# LGCN
def set_memory_growth():
    """Set if memory growth should be enabled for ALL `PhysicalDevice`.

    If memory growth is enabled for ALL `PhysicalDevice`, the runtime initialization
    will not allocate all memory on the device. Memory growth cannot be configured
    on a `PhysicalDevice` with virtual devices configured.

    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def print_table(paras):
    assert texttable, "Please install `texttable` package!"
    if not isinstance(paras, dict):
        raise TypeError("The input should be the instance of `dict`.")
    t = texttable.Texttable()
    paras = paras.copy()
    name = paras.pop('name', None)
    sorted_keys = sorted(paras.keys())
    items = [(key, str(paras[key])) for key in sorted_keys]
    t.add_rows([['Parameters', 'Value'], ['Name', name], *list(items)])
    print(t.draw())
