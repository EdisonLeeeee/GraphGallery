import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from numbers import Number
from tensorflow.keras import backend as K
from .data_utils import sparse_to_tuple

def to_tensor(inputs):
    def matrix_to_tensor(matrix):
        if any((tf.is_tensor(matrix), K.is_sparse(matrix), matrix is None)):
            return matrix
        elif sp.isspmatrix_csr(matrix) or sp.isspmatrix_csc(matrix):
            return tf.sparse.SparseTensor(*sparse_to_tuple(matrix))
        elif isinstance(matrix, (np.ndarray, list)):
            return tf.convert_to_tensor(matrix)
        else:
            raise ValueError(f'Invalid type `{type(matrix)}` of inputs data. Allowed data type (Tensor, SparseTensor, np.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix, None).')
            
    # Check `not isinstance(inputs[0], Number)` to avoid like [matrix, [1,2,3]], where [1,2,3] will be converted seperately.
    if isinstance(inputs, (list, tuple)) and not isinstance(inputs[0], Number): 
        return [to_tensor(matrix) for matrix in inputs]
    else:
        return matrix_to_tensor(inputs)