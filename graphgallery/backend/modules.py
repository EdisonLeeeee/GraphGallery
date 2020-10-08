import torch
import tensorflow as tf
from abc import ABC


class BackendModule(ABC):
    """Base Backend Module Class."""

    def __init__(self):
        ...

    @property
    def version(self) -> str:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def kind(self) -> str:
        ...

    def __str__(self) -> str:
        return f"{self.name} {self.version} Backend"

    def __repr__(self) -> str:
        return f"{self.name} {self.version} Backend"


class TensorFlowBackend(BackendModule):
    def __init__(self):
        ...

    @property
    def version(self) -> str:
        return tf.__version__

    @property
    def name(self) -> str:
        return "TensorFlow"

    @property
    def kind(self) -> str:
        return "T"


class PyTorchBackend(BackendModule):
    def __init__(self):
        ...

    @property
    def version(self) -> str:
        return torch.__version__

    @property
    def name(self) -> str:
        return "PyTorch"

    @property
    def kind(self) -> str:
        return "P"
