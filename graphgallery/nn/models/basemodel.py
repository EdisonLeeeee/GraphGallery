import os
import random
import logging

import os.path as osp
import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from graphgallery import config, check_and_convert, asintarr, Bunch
from graphgallery.utils.type_check import is_list_like
from graphgallery.utils.misc import print_table
from graphgallery.data.io import makedirs


class BaseModel:
    """Base model for semi-supervised learning and unsupervised learning."""

    def __init__(self, adj, x=None, labels=None, device="CPU:0", seed=None, name=None, **kwargs):
        """Creat an Base model for semi-supervised learning and unsupervised learning.

        Parameters:
        ----------
            adj: Scipy.sparse.csr_matrix or Numpy.ndarray, shape [n_nodes, n_nodes]
                The input `symmetric` adjacency matrix in 
                CSR format if `is_adj_sparse=True` (default)
                or Numpy format if `is_adj_sparse=False`.
            x: Scipy.sparse.csr_matrix or Numpy.ndarray, shape [n_nodes, n_attrs], optional. 
                Node attribute matrix in 
                CSR format if `is_attribute_sparse=True` 
                or Numpy format if `is_attribute_sparse=False` (default).
            labels: Numpy.ndarray, shape [n_nodes], optional
                Array, where each entry represents respective node's label(s).
            device: string. optional
                The device where the model running on.
            seed: interger scalar. optional
                Used in combination with `tf.random.set_seed` & `np.random.seed` 
                & `random.seed` to create a reproducible sequence of tensors 
                across multiple calls.
            name: string. optional
                Specified name for the model. (default: :str: `class.__name__`)
            kwargs: other customed keyword Parameters.
                `is_adj_sparse`: bool, (default: :obj: True)
                    specify the input adjacency matrix is Scipy sparse matrix or not.
                `is_attribute_sparse`: bool, (default: :obj: False)
                    specify the input attribute matrix is Scipy sparse matrix or not.
                `is_weighted`: bool, (default: :obj: False)
                    specify the input adjacency matrix is weighted or not.                    
        Note:
        ----------
            By default, `adj` is Scipy sparse matrix and `x` is Numpy array. 
                Both of them are 2-D matrices.
        """

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            tf.random.set_seed(seed)

        if name is None:
            name = self.__class__.__name__

        # By default, adj is a sparse matrix and x is a dense array (matrix)
        is_adj_sparse = kwargs.pop("is_adj_sparse", True)
        is_attribute_sparse = kwargs.pop("is_attribute_sparse", False)
        is_weighted = kwargs.pop("is_weighted", False)

        assert not is_attribute_sparse, "Node attributes matrix `x` is sparse matrix is NOT implemented"

        # leave it blank for future
        allowed_kwargs = set([])
        unknown_kwargs = set(kwargs.keys()) - allowed_kwargs
        if unknown_kwargs:
            raise ValueError(
                "Invalid keyword argument(s) in `__init__`: %s" % (unknown_kwargs,))

        # check the input adj and x, and convert them to appropriate forms
        self.n_nodes = adj.shape[0]

        self.is_adj_sparse = is_adj_sparse
        self.is_attribute_sparse = is_attribute_sparse
        self.adj, self.x = self._check_inputs(adj, x)

        if labels is not None:
            labels = asintarr(labels)
            self.n_classes = np.max(labels) + 1
        else:
            self.n_classes = None

        if x is not None:
            self.n_attributes = x.shape[1]
        else:
            self.n_attributes = None

        self.seed = seed
        self.device = device
        self.labels = labels
        self.name = name
        self.idx_train = None
        self.idx_val = None
        self.idx_test = None
        self.backup = None

        self.__model = None
        self.__custom_objects = None  # used for save/load model
        self.__sparse = is_adj_sparse

        # log path
        self.weight_dir = osp.expanduser(osp.normpath("/tmp/weight"))
        self.weight_path = osp.join(self.weight_dir, f"{name}_weights")

        # data types, default: `float32` and `int64`
        self.floatx = config.floatx()
        self.intx = config.intx()

        ############# Record paras ###########
        self.paras = Bunch(device=device, seed=seed, name=name,
                           is_adj_sparse=is_adj_sparse,
                           is_attribute_sparse=is_attribute_sparse,
                           is_weighted=is_weighted)

        self.model_paras = Bunch(name=name)
        self.train_paras = Bunch(name=name)
        ######################################

    def _check_inputs(self, adj, x):
        """Check the input adj and x and make sure they are legal inputs.

        Parameters:
        ----------
            adj: Scipy.sparse.csr_matrix or Numpy.ndarray, shape [n_nodes, n_nodes]
                The input `symmetric` adjacency matrix in 
                CSR format if `is_adj_sparse=True` (default)
                or Numpy format if `is_adj_sparse=False`.
            x: Scipy.sparse.csr_matrix or Numpy.ndarray, shape [n_nodes, n_attrs], optional. 
                Node attribute matrix in 
                CSR format if `is_attribute_sparse=True` 
                or Numpy format if `is_attribute_sparse=False` (default).

        Note:
        ----------
            By default, `adj` is Scipy sparse matrix and `x` is Numpy array. 
                Both of them are 2-D matrices.

        """
        adj = check_and_convert(adj, self.is_adj_sparse)
        x = check_and_convert(x, self.is_attribute_sparse)

        if is_list_like(adj):
            adj_shape = adj[0].shape
        else:
            adj_shape = adj.shape

        if x is not None:
            x_shape = x.shape

            if adj_shape[0] != x_shape[0]:
                raise RuntimeError(f"The first dimension of adjacency matrix and attribute matrix should be equal.")

            if len(adj_shape) != len(x_shape) != 2:
                raise RuntimeError(f"The adjacency matrix and attribute matrix should have the SAME dimensions 2.")

            if adj_shape[0] != adj_shape[1]:
                raise RuntimeError(f"The adjacency matrix should be N by N square matrix.")
        return adj, x

    def save(self, path=None, as_model=False):
        if not osp.exists(self.weight_dir):
            makedirs(self.weight_dir)
            logging.log(logging.WARNING, f"Make Directory in {self.weight_dir}")

        if path is None:
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
            model = tf.keras.models.load_model(path, custom_objects=self.__custom_objects)
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

        ```sh
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

    def __getattr__(self, attr):
        ################### TODO: This may cause ERROR #############
        try:
            return self.__dict__[attr]
        except KeyError:
            if hasattr(self.model, attr):
                return getattr(self.model, attr)
            raise AttributeError(f"'{self.name}' and '{self.name}.model' objects have no attribute '{attr}'")

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, m):
        # Back up
        if isinstance(m, tf.keras.Model) and m.weights:
            self.backup = tf.identity_n(m.weights)
        # assert m is None or isinstance(m, tf.keras.Model)
        self.__model = m

    @property
    def custom_objects(self):
        return self.__custom_objects

    @custom_objects.setter
    def custom_objects(self, value):
        assert isinstance(value, dict)
        self.__custom_objects = value

    @property
    def sparse(self):
        return self.__sparse

    @sparse.setter
    def sparse(self, value):
        self.__sparse = value
