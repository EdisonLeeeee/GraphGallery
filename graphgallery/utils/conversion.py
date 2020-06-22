import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from numbers import Number
from tensorflow.keras import backend as K

from graphgallery import config
from graphgallery.utils.type_check import is_list_like, is_interger_scalar, is_tensor_or_variable


__all__ = ['sparse_adj_to_sparse_tensor', 'sparse_tensor_to_sparse_adj', 
'sparse_adj_to_edges', 'edges_to_sparse_adj', 'to_int', 'to_tensor',
]

def sparse_adj_to_sparse_tensor(x):
    """Converts a SciPy sparse matrix to a SparseTensor."""
    sparse_coo = x.tocoo()
    row, col = sparse_coo.row, sparse_coo.col
    data, shape = sparse_coo.data, sparse_coo.shape
    if issubclass(data.dtype.type, np.floating):
        data = data.astype(config.floatx())
    indices = np.concatenate(
        (np.expand_dims(row, axis=1), np.expand_dims(col, axis=1)), axis=1)
    return tf.sparse.SparseTensor(indices, data, shape)


def sparse_tensor_to_sparse_adj(x):
    """Converts a SparseTensor to a SciPy sparse matrix."""
    # TODO
    return x

def sparse_adj_to_edges(adj):
    """Convert a Scipy sparse matrix to (edge_index, edge_weight) representation
    
    edge_index: shape [M, 2]
    edge_weight: shape [M,]
    
    """
    adj = adj.tocoo()
    edge_index = np.stack([adj.row, adj.col], axis=1).astype(config.intx())
    edge_weight = adj.data.astype(config.floatx())
    
    return edge_index, edge_weight

def edges_to_sparse_adj(edge_index, edge_weight):
    """Convert (edge_index, edge_weight) representation to a Scipy sparse matrix
    
    edge_index: shape [M, 2]
    edge_weight: shape [M,]
    
    """    
    n = np.max(edge_index) + 1
    adj = sp.csr_matrix((edge_weight, (edge_index[:,0], edge_index[:, 1])), shape=(n,n))
    return adj.astype(config.floatx(), copy=False)


def inferer_type(x):
    x = np.asarray(x)
    if x.dtype.kind == 'f':
        return config.floatx()
    elif x.dtype.kind == 'i':
        return config.intx()
    elif x.dtype.kind == 'b':
        return 'bool'
    else:
        raise RuntimeError(f'Invalid types, type `{type(x)}`')



def to_int(index):
    """Convert `index` to interger type.

    """
    if is_tensor_or_variable(index):
        return tf.cast(index, config.intx())

    if is_interger_scalar(index):
        index = np.asarray([index])
    elif is_list_like(index):
        index = np.asarray(index)
    elif isinstance(index, np.ndarray):
        pass
    else:
        raise TypeError('`index` should be either `list`, integer scalar or `np.array`!')
    return index.astype(config.intx())


def to_tensor(inputs):
    """Convert input matrices to Tensors (SparseTensors)."""
    def matrix_to_tensor(matrix):
        if any((is_tensor_or_variable(matrix), K.is_sparse(matrix), matrix is None)):
            return matrix
        elif sp.isspmatrix_csr(matrix) or sp.isspmatrix_csc(matrix):
            return sparse_adj_to_sparse_tensor(matrix)
        elif isinstance(matrix, (np.ndarray, np.matrix)) or is_list_like(matrix):
            return tf.convert_to_tensor(matrix, dtype=inferer_type(matrix))
        else:
            raise TypeError(f'Invalid type `{type(matrix)}` of inputs data. Allowed data type (Tensor, SparseTensor, np.ndarray, scipy.sparse.sparsetensor, None).')

    # Check `not isinstance(inputs[0], Number)` to avoid the situation like [1,2,3],
    # where [1,2,3] will be converted to three tensors seperately.
    if is_list_like(inputs) and not isinstance(inputs[0], Number):
        return [to_tensor(matrix) for matrix in inputs]
    else:
        return matrix_to_tensor(inputs)
