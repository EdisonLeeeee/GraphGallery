import torch
import numpy as np
import torch.nn as nn

from torch import optim
from torch.autograd import Variable

from collections import OrderedDict
from graphgallery.utils import saver


class TorchKeras(nn.Module):
    """Keras like PyTorch Model."""

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # To be compatible with TensorFlow
        self._in_multi_worker_mode = dummy_function
        self._is_graph_network = dummy_function
        self.distribute_strategy = None

        # initialize
        self.optimizer = None
        self.metrics = None
        self.loss = None
        self.metrics_names = None

    def build(self, inputs):
        # TODO
        pass

    def compile(self, loss=None,
                optimizer=None,
                metrics=None):
        self.optimizer = optimizer
        self.metrics = metrics
        self.loss = loss
        self.metrics_names = ['loss', self.metrics.name]

    def reset_metrics(self):
        assert self.metrics is not None
        self.metrics.reset_states()

    def summary(self):
        # TODO
        pass

    def save_weights(self, filepath, overwrite=True,
                     save_format=None, **kwargs):
        saver.save_torch_weights(self, filepath, overwrite=overwrite,
                                 save_format=save_format, **kwargs)

    def save(self, filepath, overwrite=True, save_format=None, **kwargs):
        saver.save_torch_model(self, filepath, overwrite=overwrite,
                               save_format=save_format, **kwargs)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()


def dummy_function(*args, **kwargs):
    ...
