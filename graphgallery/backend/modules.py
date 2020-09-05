import torch
import tensorflow as tf
from abc import ABC


class BackendModule(ABC):
    """Base Backend Module Class."""

    def __init__(self):
        ...

    @property
    def version(self):
        ...

    @property
    def name(self):
        ...

    @property
    def kind(self):
        ...

    def __str__(self):
        return f"{self.name} {self.version} Backend"

    def __repr__(self):
        return f"{self.name} {self.version} Backend"


class TensorFlowBackend(BackendModule):
    def __init__(self):
        ...

    @property
    def version(self):
        return tf.__version__

    @property
    def name(self):
        return "TensorFlow"

    @property
    def kind(self):
        return "T"


class PyTorchBackend(BackendModule):
    def __init__(self):
        ...

    @property
    def version(self):
        return torch.__version__

    @property
    def name(self):
        return "PyTorch"

    @property
    def kind(self):
        return "P"
