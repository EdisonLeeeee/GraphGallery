import random
import torch

import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from abc import ABC

from graphgallery import intx, floatx, backend, set_backend, is_list_like
from graphgallery.data import Basegraph, Graph
from graphgallery.utils.raise_error import raise_if_kwargs
from graphgallery.utils.device import parse_device


# def _check_cur_module(module, kind):
#     modules = module.split('.')[-4:]
#     if any(("tf_models" in modules and kind == "P",
#             "th_models" in modules and kind == "T")):
#         cur_module = "Tensorflow models" if kind == "P" else "PyTorch models"
#         raise RuntimeError(f"You are currently using models in '{cur_module}' but with backend '{backend()}'."
#                            "Please use `set_backend()` to change the current backend.")


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



class Base(ABC):
    """Base model for all class."""

    def __init__(self, *graph, device="cpu:0", seed=None, name=None, **kwargs):
        """Create an Base model for semi-supervised learning and unsupervised learning.

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
            kwargs: other customized keyword Parameters.

        """
        graph = parse_graph_inputs(*graph)
        self.backend = backend()
        self.kind = self.backend.kind

        raise_if_kwargs(kwargs)

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            if self.kind == "P":
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                # torch.cuda.manual_seed_all(seed)
            else:
                tf.random.set_seed(seed)

        if name is None:
            name = self.__class__.__name__

        self.graph = graph.copy()
        self.seed = seed
        self.name = name
        self.device = parse_device(device, self.kind)

        # data types, default: `float32` and `int32`
        self.floatx = floatx()
        self.intx = intx()

