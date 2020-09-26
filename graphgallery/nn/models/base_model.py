import os
import random
import logging
import torch

import numpy as np
import tensorflow as tf
import os.path as osp
import scipy.sparse as sp

from tensorflow.keras import backend as K

from abc import ABC

from graphgallery import intx, floatx, backend, set_backend, is_list_like
from graphgallery.data.io import makedirs_from_path
from graphgallery.data import Basegraph, Graph
from graphgallery.utils.raise_error import raise_if_kwargs
from graphgallery.utils import save


def _check_cur_module(module, kind):
    modules = module.split('.')[-4:]
    if any(("TF" in modules and kind == "P",
            "PTH" in modules and kind == "T")):
        cur_module = "Tensorflow models" if kind == "P" else "PyTorch models"
        raise RuntimeError(f"You are currently using models in '{cur_module}' but with backend '{backend()}'."
                           "Please use `set_backend()` to change the current backend.")


def parse_graph_inputs(*graph):
    # TODO: Maybe I could write it a little more elegantly here?
    if len(graph) == 0:
        graph = None
    elif len(graph) == 1:
        graph, = graph
        if isinstance(graph, Basegraph):
            ...
        elif sp.isspmatrix(graph):
            graph = Graph(graph)
        elif isinstance(graph, dict):
            return Graph(**graph)
        elif is_list_like(graph):
            # TODO: multi graph
            ...
        else:
            raise ValueError(f"Unrecognized inputs {graph}.")
    else:
        if sp.isspmatrix(graph[0]):
            graph = Graph(*graph)
        elif is_list_like(graph[0]):
            # TODO: multi graph
            ...
        else:
            raise ValueError(f"Unrecognized inputs {graph}.")

    return graph


def parse_device(device: str, kind: str) -> str:
    # TODO:
    # 1. device can be torch.device
    # 2. check if gpu is available
    _device = osp.split(device.lower())[1]
    if not any((_device.startswith("cpu"),
                _device.startswith("cuda"),
                _device.startswith("gpu"))):
        raise RuntimeError(
            f" Expected one of cpu (CPU), cuda (CUDA), gpu (GPU) device type at start of device string: {device}")

    if _device.startswith("cuda"):
        if kind == "T":
            _device = "GPU" + _device[4:]
    elif _device.startswith("gpu"):
        if kind == "P":
            _device = "cuda" + _device[3:]

    if kind == "P":
        if _device.startswith('cuda'):
            torch.cuda.empty_cache()
        return torch.device(_device)
    return _device


class BaseModel(ABC):
    """Base model for semi-supervised learning and unsupervised learning."""

    def __init__(self, *graph, device="cpu:0", seed=None, name=None, **kwargs):
        """Creat an Base model for semi-supervised learning and unsupervised learning.

        Parameters:
        ----------
            graph: Graph or MultiGraph.
            device: string. optional
                The device where the model running on.
            seed: interger scalar. optional
                Used in combination with `tf.random.set_seed` & `np.random.seed`
                & `random.seed` to create a reproducible sequence of tensors
                across multiple calls.
            name: string. optional
                Specified name for the model. (default: :str: `class.__name__`)
            kwargs: other customed keyword Parameters.

        """
        graph = parse_graph_inputs(*graph)
        self.backend = backend()
        self.kind = self.backend.kind

        if kwargs.pop('check', True):
            _check_cur_module(self.__module__, self.kind)

        _id = np.random.RandomState(None).randint(100)

        raise_if_kwargs(kwargs)

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            if self.kind == "P":
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
#                 torch.cuda.manual_seed_all(seed)
            else:
                tf.random.set_seed(seed)

        if name is None:
            name = self.__class__.__name__

        self.graph = graph.copy()
        self.seed = seed
        self.name = name
        self.device = parse_device(device, self.kind)
        self.idx_train = None
        self.idx_val = None
        self.idx_test = None
        self.backup = None

        self._model = None
        self._custom_objects = None  # used for save/load TF model

        # log path
        # add random integer to avoid duplication
        self.weight_path = osp.join(osp.expanduser(osp.normpath("/tmp/weight")),
                                    f"{name}_{_id}_weights")

        # data types, default: `float32` and `int32`
        self.floatx = floatx()
        self.intx = intx()

    def save(self, path=None, as_model=False, overwrite=True, save_format=None, **kwargs):

        if not path:
            path = self.weight_path

        makedirs_from_path(path)

        if as_model:
            if self.kind == "T":
                save.save_tf_model(self.model, path, overwrite=overwrite, save_format=save_format, **kwargs)
            else:
                save.save_torch_model(self.model, path, overwrite=overwrite, save_format=save_format, **kwargs)
        else:
            if self.kind == "T":
                save.save_tf_weights(self.model, path, overwrite=overwrite, save_format=save_format)
            else:
                save.save_torch_weights(self.model, path, overwrite=overwrite, save_format=save_format)

    def load(self, path=None, as_model=False):
        if not path:
            path = self.weight_path

        if as_model:
            if self.kind == "T":
                self.model = save.load_tf_model(
                    path, custom_objects=self.custom_objects)
            else:
                self.model = save.load_torch_model(path)
        else:
            if self.kind == "T":
                save.load_tf_weights(self.model, path)
            else:
                save.load_torch_weights(self.model, path)

    def __getattr__(self, attr):
        ##### TODO: This may cause ERROR ######
        try:
            return self.__dict__[attr]
        except KeyError:
            if hasattr(self, "_model") and hasattr(self._model, attr):
                return getattr(self._model, attr)
            raise AttributeError(
                f"'{self.name}' and '{self.name}.model' objects have no attribute '{attr}'")

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, m):
        # Back up
        if isinstance(m, tf.keras.Model) and m.weights:
            self.backup = tf.identity_n(m.weights)
        # assert m is None or isinstance(m, tf.keras.Model) or torch.nn.Module
        self._model = m

    @property
    def custom_objects(self):
        return self._custom_objects

    @custom_objects.setter
    def custom_objects(self, value):
        assert isinstance(value, dict)
        self._custom_objects = value

    @property
    def close(self):
        """Close the session of model and set `built` to False."""
        K.clear_session()
        self.model = None

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def __repr__(self):
        return f"GraphGallery.nn.{self.name}(device={self.device}, backend={self.backend})"
