import os
import random
import logging

import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from graphgallery import config
from graphgallery.utils import conversion


def _check_adj(adj, is_adj_sparse):
    # By default, adj is sparse matrix
    if is_adj_sparse:
        if not (sp.isspmatrix_csr(adj) or sp.isspmatrix_csc(adj)):
            raise TypeError("Adjacency matrix `adj` must be `scipy.sparse.csr_matrix`"
                               f" or `csc_matrix` when `is_adj_sparse=True`, but got {type(adj)}")
    else:
        if not isinstance(adj, (np.ndarray, np.matrix)):
            raise TypeError("Adjacency matrix `adj` must be `np.array` or `np.matrix`"
                               f" when `is_adj_sparse=False`, but got {type(adj)}") 
        return np.asarray(adj, dtype=config.floatx())            
            
    return adj.astype(dtype=config.floatx(), copy=False)


def _check_x(x, is_x_sparse):
    # By default, x is not sparse matrix
    if not is_x_sparse:
        if not isinstance(x, (np.ndarray, np.matrix)):
            raise TypeError("Feature matrix `x` must be `np.array` or `np.matrix`"
                               f" when `is_x_sparse=False`, but got {type(x)}") 
        return np.asarray(x, dtype=config.floatx())
    else:
        if not not (sp.isspmatrix_csr(x) or sp.isspmatrix_csc(x)):
            raise TypeError("Feature matrix `x` must be `scipy.sparse.csr_matrix`"
                               f" or `csc_matrix` when `is_x_sparse=True`, but got {type(x)}")   
    return x.astype(dtype=config.floatx(), copy=False)


def _check_inputs(adj, x, is_adj_sparse, is_x_sparse):
    """Check the input adj and x and make sure they are legal forms of input.

    Arguments:
    ----------
        adj: shape (N, N), `scipy.sparse.csr_matrix` (or `csc_matrix`) if 
            `is_adj_sparse=True`, `np.array` or `np.matrix` if `is_adj_sparse=False`.
            The input `symmetric` adjacency matrix, where `N` is the number 
            of nodes in graph.
        x: shape (N, F), `scipy.sparse.csr_matrix` (or `csc_matrix`) if 
            `is_x_sparse=True`, `np.array` or `np.matrix` if `is_x_sparse=False`.
            The input node feature matrix, where `F` is the dimension of features.

    Note:
    ----------
        By default, `adj` is sparse matrix and `x` is dense array. Both of them are 
        2-D matrices.


    """
    adj = _check_adj(adj, is_adj_sparse)
    x = _check_x(x, is_x_sparse)
    adj_shape = adj.shape
    x_shape = x.shape
    
    if adj_shape[0] != x_shape[0]:
        raise RuntimeError(f"The first dimension of adjacency matrix and feature matrix should be equal.")

    if len(adj_shape) != len(x_shape) != 2:
        raise RuntimeError(f"The adjacency matrix and feature matrix should have the SAME dimensions 2.")
        
    if adj_shape[0] != adj_shape[1]:
        raise RuntimeError(f"The adjacency matrix should be N by N square matrix.")        
    return adj, x


class BaseModel:
    """Base model for supervised learning and unsupervised learning.


    Arguments:
    ----------
        adj: shape (N, N), `scipy.sparse.csr_matrix` (or `csc_matrix`) if 
            `is_adj_sparse=True`, `np.array` or `np.matrix` if `is_adj_sparse=False`.
            The input `symmetric` adjacency matrix, where `N` is the number 
            of nodes in graph.
        x: shape (N, F), `scipy.sparse.csr_matrix` (or `csc_matrix`) if 
            `is_x_sparse=True`, `np.array` or `np.matrix` if `is_x_sparse=False`.
            The input node feature matrix, where `F` is the dimension of features.
        labels: shape (N,), array-like. Default: `None` for unsupervised learning.
            The class labels of the nodes in the graph. 
        device: string. Default: `CPU:0`.
            The device where the model running on.
        seed: interger scalar. Default: `None`.
            Used in combination with `tf.random.set_seed` & `np.random.seed` & `random.seed`  
            to create a reproducible sequence of tensors across multiple calls. 
        name (String, optional): 
                Specified name for the model. (default: `class.__name__`)
        kwargs: other customed keyword arguments.
            
    Note:
    ----------
        By default, `adj` is sparse matrix and `x` is dense array. Both of them are 
        2-D matrices.

    """

    def __init__(self, adj, x, labels=None, device="CPU:0", seed=None, name=None, **kwargs):

        if seed:
            tf.random.set_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        if name is None:
            name = self.__class__.__name__

        # By default, adj is sparse matrix and x is dense array (matrix)
        is_adj_sparse = kwargs.pop("is_adj_sparse", True)
        is_x_sparse = kwargs.pop("is_x_sparse", False)
        is_weighted = kwargs.pop("is_weighted", False)

        # leave it blank for the future
        allowed_kwargs = set([])
        unknown_kwargs = set(kwargs.keys()) - allowed_kwargs
        if unknown_kwargs:
            raise TypeError(
                "Invalid keyword argument(s) in `__init__`: %s" % (unknown_kwargs,))

        # check the input adj and x, and convert them to appropriate forms
        self.adj, self.x = _check_inputs(adj, x, is_adj_sparse, is_x_sparse)
        self.n_nodes, self.n_features = x.shape

        self.is_adj_sparse = is_adj_sparse
        self.is_x_sparse = is_x_sparse

        if labels is not None:
            self.n_classes = np.max(labels) + 1
        else:
            self.n_classes = None

        self.seed = seed
        self.device = device
        self.labels = self.to_int(labels)
        self.name = name
        self._model = None
        self.built = None
        self.index_train = None
        self.index_val = None
        self.index_test = None
        self.do_before_train = None
        self.do_before_validation = None
        self.do_before_test = None
        self.do_before_predict = None
        self.sparse = True
        self.norm_x_fn = None
        self.custom_objects = None  # used for save/load model

        self.log_path = f"./log/{name}_weights"

        # data types, default: `float32` and `int64`
        self.floatx = config.floatx()
        self.intx = config.intx()

    @property
    def model(self):
        return self._model

    def set_model(self, model):
        self._model = model

    def preprocess(self, adj, x):
        """Preprocess the input adjacency matrix and feature matrix, e.g., normalization.
        And convert them to tf.tensor. By default, the adj and x will not be
        preprocessed.

        Arguments:
        ----------
            adj: shape (N, N), `scipy.sparse.csr_matrix` (or `csc_matrix`) if 
                `is_adj_sparse=True`, `np.array` if `is_adj_sparse=False`.
                The input `symmetric` adjacency matrix, where `N` is the number 
                of nodes in graph.
            x: shape (N, F), `scipy.sparse.csr_matrix` (or `csc_matrix`) if 
                `is_x_sparse=True`, `np.array` if `is_x_sparse=False`.
                The input node feature matrix, where `F` is the dimension of node features.
                
        Note:
        ----------
            By default, `adj` is sparse matrix and `x` is dense array. Both of them are 
            2-D matrices.
        """
        adj, x = _check_inputs(adj, x, self.is_adj_sparse, self.is_x_sparse)
        self.n_nodes, self.n_features = x.shape

        return adj, x

    @staticmethod
    def to_tensor(inputs):
        """Convert input matrices to Tensor (SparseTensor)."""
        return conversion.to_tensor(inputs)

    @staticmethod
    def to_int(inputs):
        return conversion.to_int(inputs)

    def save(self, path=None, save_model=False):
        if not os.path.exists("log"):
            os.makedirs("log")
            logging.log(logging.WARNING, "Mkdir /log")

        if path is None:
            path = self.log_path
            
        if not path.endswith('.h5'):
            path += '.h5' 
            
        if save_model:
            self.model.save(path, save_format="h5")
        else:
            try:
                self.model.save_weights(path)
            except ValueError as e:
                # due to the bugs in tf 2.1
                self.model.save_weights(path[:-3])


    def load(self, path=None, save_model=False):
        if path is None:
            path = self.log_path
        if not path.endswith('.h5'):
            path += '.h5'      
        if save_model:
            model = tf.keras.models.load_model(path, custom_objects=self.custom_objects)
            self.set_model(model)
        else:
            try:
                self.model.load_weights(path)
            except KeyError as e:
                self.model.load_weights(path[:-3])

    def __repr__(self):
        return f"Graphgallery.nn.{self.name} in {self.device}"
