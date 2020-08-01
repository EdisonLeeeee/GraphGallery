import os
import random
import logging

import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from graphgallery import config, check_and_convert, asintarr, Bunch
from graphgallery.utils.type_check import is_list_like
from graphgallery.utils.misc import print_table


class BaseModel:
    """Base model for supervised learning and unsupervised learning.


    Arguments:
    ----------
        adj: shape (N, N), `scipy.sparse.csr_matrix` (or `csc_matrix`) if 
            `is_adj_sparse=True`, Numpy array-like or Numpy matrix if `is_adj_sparse=False`.
            The input `symmetric` adjacency matrix, where `N` is the number 
            of nodes in graph.
        x: shape (N, F), `scipy.sparse.csr_matrix` (or `csc_matrix`) if 
            `is_x_sparse=True`, Numpy array-like or Numpy matrix if `is_x_sparse=False`.
            The input node feature matrix, where `F` is the dimension of features.
        labels: shape (N,), array-like. Default: `None` for unsupervised learning.
            The class labels of the nodes in the graph. 
        device: string. Default: `CPU:0`.
            The device where the model running on.
        seed: interger scalar. Default: `None`.
            Used in combination with `tf.random.set_seed` & `np.random.seed` & `random.seed`  
            to create a reproducible sequence of tensors across multiple calls. 
        name (String, optional): 
                Specified name for the model. (default: :str: `class.__name__`)
        kwargs: other customed keyword arguments.

    Note:
    ----------
        By default, `adj` is sparse matrix and `x` is dense array. Both of them are 
        2-D matrices.

    """

    def __init__(self, adj, x, labels=None, device="CPU:0", seed=None, name=None, **kwargs):

        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        if name is None:
            name = self.__class__.__name__

        # By default, adj is sparse matrix and x is dense array (matrix)
        is_adj_sparse = kwargs.pop("is_adj_sparse", True)
        is_x_sparse = kwargs.pop("is_x_sparse", False)
        is_weighted = kwargs.pop("is_weighted", False)

        # leave it blank for future
        allowed_kwargs = set([])
        unknown_kwargs = set(kwargs.keys()) - allowed_kwargs
        if unknown_kwargs:
            raise ValueError(
                "Invalid keyword argument(s) in `__init__`: %s" % (unknown_kwargs,))

        # check the input adj and x, and convert them to appropriate forms
        self.n_nodes, self.n_features = x.shape

        self.is_adj_sparse = is_adj_sparse
        self.is_x_sparse = is_x_sparse
        self.adj, self.x = self._check_inputs(adj, x)

        if labels is not None:
            self.n_classes = np.max(labels) + 1
        else:
            self.n_classes = None

        self.seed = seed
        self.device = device
        self.labels = asintarr(labels)
        self.name = name
        self.__model = None
        self.idx_train = None
        self.idx_val = None
        self.idx_test = None
        self.do_before_train = None
        self.do_before_validation = None
        self.do_before_test = None
        self.do_before_predict = None
        self.sparse = True
        self.norm_x_fn = None
        self.custom_objects = None  # used for save/load model

        self.weight_path = f"./weight/{name}_weights"

        # data types, default: `float32` and `int64`
        self.floatx = config.floatx()
        self.intx = config.intx()
        # Paraneters
        self.paras = Bunch(device=device, seed=seed, name=name,
                           is_adj_sparse=is_adj_sparse,
                           is_x_sparse=is_x_sparse,
                           is_weighted=is_weighted)

        self.model_paras = Bunch(name=name)
        self.train_paras = Bunch(name=name)

    def _check_inputs(self, adj, x):
        """Check the input adj and x and make sure they are legal forms of input.

        Arguments:
        ----------
            adj: shape (N, N), `scipy.sparse.csr_matrix` (or `csc_matrix`) if 
                `is_adj_sparse=True`, Numpy array-like or Numpy matrix if `is_adj_sparse=False`.
                The input `symmetric` adjacency matrix, where `N` is the number 
                of nodes in graph.
            x: shape (N, F), `scipy.sparse.csr_matrix` (or `csc_matrix`) if 
                `is_x_sparse=True`, Numpy array-like or Numpy matrix if `is_x_sparse=False`.
                The input node feature matrix, where `F` is the dimension of features.

        Note:
        ----------
            By default, `adj` is sparse matrix and `x` is dense array. Both of them are 
            2-D matrices.


        """
        adj = check_and_convert(adj, self.is_adj_sparse)
        x = check_and_convert(x, self.is_x_sparse)
        if is_list_like(adj):
            adj_shape = adj[0].shape
        else:
            adj_shape = adj.shape

        x_shape = x.shape

        if adj_shape[0] != x_shape[0]:
            raise RuntimeError(f"The first dimension of adjacency matrix and feature matrix should be equal.")

        if len(adj_shape) != len(x_shape) != 2:
            raise RuntimeError(f"The adjacency matrix and feature matrix should have the SAME dimensions 2.")

        if adj_shape[0] != adj_shape[1]:
            raise RuntimeError(f"The adjacency matrix should be N by N square matrix.")
        return adj, x

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, m):
#         assert m is None or isinstance(m, tf.keras.Model)
        self.__model = m

    def preprocess(self, adj, x):
        """Preprocess the input adjacency matrix and feature matrix, e.g., normalization.
        And convert them to tf.tensor. 

        Arguments:
        ----------
            adj: shape (N, N), `scipy.sparse.csr_matrix` (or `csc_matrix`) if 
                `is_adj_sparse=True`, Numpy array-like if `is_adj_sparse=False`.
                The input `symmetric` adjacency matrix, where `N` is the number 
                of nodes in graph.
            x: shape (N, F), `scipy.sparse.csr_matrix` (or `csc_matrix`) if 
                `is_x_sparse=True`, Numpy array-like if `is_x_sparse=False`.
                The input node feature matrix, where `F` is the dimension of node features.

        Note:
        ----------
            By default, `adj` is sparse matrix and `x` is dense array. Both of them are 
            2-D matrices.
        """
        # check the input adj and x, and convert them to appropriate forms
        self.adj, self.x = self._check_inputs(adj, x)
        self.n_nodes, self.n_features = x.shape

    def save(self, path=None, as_model=False):
        if not os.path.exists("weight"):
            os.makedirs("weight")
            logging.log(logging.WARNING, "Mkdir /weight")

        if not path:
            path = self.weight_path

        if not path.endswith('.h5'):
            path += '.h5'

        if as_model:
            self.model.save(path, save_format="h5")
        else:
            try:
                self.model.save_weights(path)
            except ValueError as e:
                # due to the bugs in tf 2.1
                self.model.save_weights(path[:-3])

    def load(self, path=None, as_model=False):
        if not path:
            path = self.weight_path
        if not path.endswith('.h5'):
            path += '.h5'
        if as_model:
            model = tf.keras.models.load_model(path, custom_objects=self.custom_objects)
            self.model = model
        else:
            try:
                self.model.load_weights(path)
            except KeyError as e:
                self.model.load_weights(path[:-3])

    def __repr__(self):
        return f"Graphgallery.nn.{self.name} in {self.device}"

    def show(self, name=None):
        """Show the parameters in a table.
        Note: You must install `texttable` package first. Using
        ```
        pip install texttable
        ```

        """
        if name == 'train':
            paras = self.train_paras
        elif name == 'model':
            paras = self.model_paras
        else:
            paras = self.paras
        print_table(paras)
