import random
import torch

import numpy as np
import tensorflow as tf
import scipy.sparse as sp
import graphgallery as gg

from abc import ABC


def parse_graph_inputs(*graph):
    # TODO: Maybe I could write it a little more elegantly here?
    if len(graph) == 0:
        graph = None
    elif len(graph) == 1:
        graph, = graph
        if isinstance(graph, gg.data.BaseGraph):
            ...
        elif sp.isspmatrix(graph):
            graph = gg.Graph(graph)
        elif isinstance(graph, dict):
            return gg.Graph(**graph)
        elif gg.is_listlike(graph):
            # TODO: multi graph
            ...
        else:
            raise ValueError(f"Unrecognized inputs {graph}.")
    else:
        if sp.isspmatrix(graph[0]):
            graph = gg.Graph(*graph)
        elif gg.is_listlike(graph[0]):
            # TODO: multi graph
            ...
        else:
            raise ValueError(f"Unrecognized inputs {graph}.")

    return graph


class Model(ABC):
    def __init__(self, *graph, device="cpu:0", seed=None, name=None, **kwargs):
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
            kwargs: other custom keyword parameters.

        """
        graph = parse_graph_inputs(*graph)
        _backend = gg.backend()

        gg.utils.raise_error.raise_if_kwargs(kwargs)

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            if _backend == "torch":
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                # torch.cuda.manual_seed_all(seed)
            else:
                tf.random.set_seed(seed)

        if name is None:
            name = self.__class__.__name__

        self.seed = seed
        self.name = name
        self.graph = graph.copy()
        self.device = gg.functional.parse_device(device, _backend)
        self.backend = _backend

        # data types, default: `float32`,`int32` and `bool`
        self.floatx = gg.floatx()
        self.intx = gg.intx()
        self.boolx = gg.boolx()
