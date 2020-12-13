import random
import torch

import numpy as np
import tensorflow as tf
import scipy.sparse as sp
import graphgallery as gg
from graphgallery import functional as gf


class Model:
    def __init__(self, graph, device="cpu", seed=None, name=None, **kwargs):
        """

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
        kwargs: other custom keyword arguments. 
        """
        if not isinstance(graph, gg.data.BaseGraph):
            raise ValueError(f"Unrecognized graph: {graph}.")

        _backend = gg.backend()

        # It currently takes no keyword arguments
        gg.utils.raise_error.raise_if_kwargs(kwargs)

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            if _backend == "tensorflow":
                tf.random.set_seed(seed)
            else:
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                # torch.cuda.manual_seed_all(seed)

        if name is None:
            name = self.__class__.__name__

        self.seed = seed
        self.name = name
        self.graph = graph.copy()
        self.device = gf.device(device, _backend)
        self.backend = _backend

        # data types, default: `float32`,`int32` and `bool`
        self.floatx = gg.floatx()
        self.intx = gg.intx()
        self.boolx = gg.boolx()
        self._cache = gf.BunchDict()

    @property
    def cache(self):
        return self._cache

    def register_cache(self, name, value):
        self._cache[name] = value

    def __repr__(self):
        return f"{self.name}(device={self.device}, backend={self.backend})"

    __str__ = __repr__
