import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from numbers import Number
from tensorflow.keras import backend as K

from graphgallery import config
from graphgallery.utils.type_check import is_list_like, is_interger_scalar, is_tensor_or_variable


__all__ = ['check_and_convert', 'sparse_adj_to_sparse_tensor', 'sparse_tensor_to_sparse_adj',
           'sparse_adj_to_edges', 'edges_to_sparse_adj', 'asintarr', 'astensor',
           ]


def check_and_convert(matrix, is_sparse):
    """Check the input matrix and convert it into a proper data type.

    Arguments:
    ----------
        matrix: Scipy sparse matrix or Numpy array-like or Numpy matrix.
        is_sparse: Indicating whether the input matrix is sparse matrix or not.

    Returns:
    ----------
        A converted matrix with proper data type.

    """
    if is_list_like(matrix):
        return [check_and_convert(m, is_sparse) for m in matrix]
    if not is_sparse:
        if not isinstance(matrix, (np.ndarray, np.matrix)):
            raise TypeError("The input matrix must be Numpy array-like or `np.matrix`"
                            f" when `is_sparse=False`, but got {type(x)}")
        return np.asarray(matrix, dtype=config.floatx())
    else:
        if not sp.issparse(matrix):
            raise TypeError(f"The input matrix must be Scipy sparse matrix when `is_sparse=True`, but got {type(matrix)}")

        return matrix.astype(dtype=config.floatx(), copy=False)


def sparse_adj_to_sparse_tensor(x):
    """Converts a Scipy sparse matrix to a SparseTensor."""
    sparse_coo = x.tocoo()
    row, col = sparse_coo.row, sparse_coo.col
    data, shape = sparse_coo.data.astype(config.floatx()), sparse_coo.shape
    indices = np.concatenate(
        (np.expand_dims(row, axis=1), np.expand_dims(col, axis=1)), axis=1)
    return tf.sparse.SparseTensor(indices, data, shape)


def sparse_tensor_to_sparse_adj(x):
    """Converts a SparseTensor to a Scipy sparse matrix (CSR matrix)."""
    data = x.values.astype(config.floatx())
    indices = x.indices.numpy().T
    shape = x.shape
    return sp.csr_matrix((data, indices), shape=shape)


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
    adj = sp.csr_matrix((edge_weight, (edge_index[:, 0], edge_index[:, 1])), shape=(n, n))
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


def asintarr(matrix, dtype=config.intx()):
    """Convert `matrix` to interger data type.

    """
    if is_tensor_or_variable(matrix):
        return tf.cast(matrix, dtype=dtype)

    if is_interger_scalar(matrix):
        matrix = np.asarray([matrix], dtype=dtype)
    elif is_list_like(matrix) or isinstance(matrix, (np.ndarray, np.matrix)):
        matrix = np.asarray(matrix, dtype=dtype)
    else:
        raise TypeError(f'Invalid input matrix which should be either array-like or integer scalar, but got {type(matrix)}.')
    return matrix


def astensor(inputs):
    """Convert input matrices to Tensors (SparseTensors).

    inputs: single or a list of array-like variables.
    """
    def matrix_astensor(matrix):
        if any((is_tensor_or_variable(matrix), K.is_sparse(matrix), matrix is None)):
            return matrix
        elif sp.isspmatrix_csr(matrix) or sp.isspmatrix_csc(matrix):
            return sparse_adj_to_sparse_tensor(matrix)
        elif isinstance(matrix, (np.ndarray, np.matrix)) or is_list_like(matrix):
            return tf.convert_to_tensor(matrix, dtype=inferer_type(matrix))
        else:
            raise TypeError(f'Invalid type `{type(matrix)}` of inputs data. Allowed data type `(Tensor, SparseTensor, numpy array, scipy sparse tensor, None)`, but got {type(matrix)}.')

    # Check `not isinstance(inputs[0], Number)` to avoid the situation like [1,2,3],
    # where [1,2,3] will be converted to three tensors seperately.
    if is_list_like(inputs) and not isinstance(inputs[0], Number):
        return [astensor(matrix) for matrix in inputs]
    else:
        return matrix_astensor(inputs)
