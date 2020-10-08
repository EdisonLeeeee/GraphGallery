"""Types for typing functions signatures."""

from typing import Union, Callable, List, Tuple, Optional, TypeVar, Generic

import torch
import numpy as np
import networkx as nx
import tensorflow as tf
import scipy.sparse as sp
from tensorflow.python.eager.context import _EagerDeviceContext

TransformType= TypeVar('TransformType')
GraphType = TypeVar('GraphType')

IntNumber = Union[
    int,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]

FloatNumber = Union[
    float,
    np.float16,
    np.float32,
    np.float64,
]

Number = Union[
    IntNumber, FloatNumber
]

Shape = Optional[Tuple[IntNumber, IntNumber]]

Device = Optional[Union[str, torch.device, _EagerDeviceContext]]

Initializer = Optional[Union[dict, str, Callable]]
Regularizer = Optional[Union[dict, str, Callable]]
Constraint = Optional[Union[dict, str, Callable]]
Activation = Optional[Union[str, Callable]]
Optimizer = Union[tf.keras.optimizers.Optimizer, str]

SparseMatrix = Union[sp.csr_matrix, sp.csc_matrix, sp.coo_matrix]
MultiSparseMatrix = Union[List[SparseMatrix], Tuple[SparseMatrix]]
ArrayLike2D = Union[List[List], np.ndarray, np.matrix]
ArrayLike1D = Union[List, np.ndarray, np.matrix]
ArrayLike = Union[ArrayLike1D, ArrayLike2D]
MultiArrayLike = Union[List[ArrayLike], Tuple[ArrayLike]]

Edge = Union[List[List], ArrayLike2D]


AcceptableTransform = Optional[Union[TransformType, str, Callable]]
# AcceptableTransform = Optional[Union[TransformType, str, Callable]]

ListLike = Union[List, Tuple]

TFTensor = Union[
    tf.Tensor,
    tf.sparse.SparseTensor,
    tf.Variable,
    tf.RaggedTensor
]

TorchTensor = torch.Tensor
TensorLike = Union[List[Union[Number, list]],
                   tuple,
                   Number,
                   ArrayLike,
                   TFTensor,
                   TorchTensor]


NxGraph = Union[nx.Graph, nx.DiGraph]

FloatTensorLike = Union[tf.Tensor, float, np.float16, np.float32, np.float64]
AcceptableDTypes = Union[tf.DType, np.dtype, type, int, str, None]
