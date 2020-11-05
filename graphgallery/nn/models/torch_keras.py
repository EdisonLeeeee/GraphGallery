import torch
import numpy as np
import torch.nn as nn

from torch.nn import Module
from torch import optim
from torch.autograd import Variable

from collections import OrderedDict
from graphgallery.utils import saver


class TorchKeras(Module):
    """Keras like PyTorch Model."""

    def __init__(self, *args, **kwargs):
        """
        Initialize the worker.

        Args:
            self: (todo): write your description
        """

        super().__init__(*args, **kwargs)

        # To be compatible with TensorFlow
        self._in_multi_worker_mode = dummy_function
        self._is_graph_network = dummy_function

    def build(self, inputs):
        """
        Builds the graph.

        Args:
            self: (todo): write your description
            inputs: (array): write your description
        """
        # TODO
        pass

    def compile(self, loss=None,
                optimizer=None,
                metrics=None):
        """
        Compile the loss.

        Args:
            self: (todo): write your description
            loss: (todo): write your description
            optimizer: (todo): write your description
            metrics: (str): write your description
        """
        # TODO
        pass

    def summary(self):
        """
        Returns the summary

        Args:
            self: (todo): write your description
        """
        # TODO
        pass

    def save_weights(self, filepath, overwrite=True, save_format=None):
        """
        Save the weights to a file.

        Args:
            self: (todo): write your description
            filepath: (str): write your description
            overwrite: (bool): write your description
            save_format: (str): write your description
        """
        saver.save_torch_weights(self, filepath, overwrite=overwrite, save_format=save_format)

    def save(self, filepath, overwrite=True, save_format=None, **kwargs):
        """
        Saves the model to a file.

        Args:
            self: (todo): write your description
            filepath: (str): write your description
            overwrite: (bool): write your description
            save_format: (str): write your description
        """
        saver.save_torch_model(self, filepath, overwrite=overwrite, save_format=save_format, **kwargs)

    def reset_parameters(self):
        """
        Reset all the parameters.

        Args:
            self: (todo): write your description
        """
        for layer in self.layers:
            layer.reset_parameters()


def dummy_function(*args, **kwargs):
    """
    Dummy function.

    Args:
    """
    ...
