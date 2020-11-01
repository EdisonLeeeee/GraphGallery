import torch
import tensorflow
import numpy


__all__ = ['BackendModule', 'TensorFlowBackend', 'PyTorchBackend']


class BackendModule:
    """Base Backend Module Class."""

    acceptable_names = set()

    def __init__(self):
        ...

    @property
    def version(self) -> str:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def abbr(self) -> str:
        return NotImplementedError

    @property
    def kind(self) -> str:
        return NotImplementedError

    def __eq__(self, value) -> bool:
        return str(value).lower() in self.acceptable_names

    def __str__(self) -> str:
        return f"{self.name} {self.version} Backend"

    def __repr__(self) -> str:
        return f"{self.name} {self.version} Backend"


class TensorFlowBackend(BackendModule):
    acceptable_names = {"t", "tf", "tensorflow"}

    def __init__(self):
        super().__init__()

    @property
    def version(self) -> str:
        return tensorflow.__version__

    @property
    def name(self) -> str:
        return "TensorFlow"

    @property
    def abbr(self) -> str:
        return "tf"

    @property
    def kind(self) -> str:
        return "t"


class PyTorchBackend(BackendModule):
    acceptable_names = {"p", "th", "torch", "pytorch"}

    def __init__(self):
        super().__init__()

    @property
    def version(self) -> str:
        return torch.__version__

    @property
    def name(self) -> str:
        return "PyTorch"

    @property
    def abbr(self) -> str:
        return "th"

    @property
    def kind(self) -> str:
        return "p"


class TensorFlowBackend(BackendModule):
    acceptable_names = {"t", "tf", "tensorflow"}

    def __init__(self):
        super().__init__()

    @property
    def version(self) -> str:
        return tensorflow.__version__

    @property
    def name(self) -> str:
        return "TensorFlow"

    @property
    def abbr(self) -> str:
        return "tf"

    @property
    def kind(self) -> str:
        return "t"
