import torch
import tensorflow
import numpy


__all__ = ['BackendModule', 'TensorFlowBackend',
           'PyTorchBackend', 'PyGBackend', 'DGLPyTorchBackend', 'DGLTensorFlowBackend']


class BackendModule:
    """Base Backend Module Class."""

    alias = {}

    def __init__(self):
        self.acceptable_names = self.alias

    @property
    def version(self) -> str:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def abbr(self) -> str:
        return NotImplementedError

    def __eq__(self, value) -> bool:
        return str(value).lower() in self.acceptable_names

    def __repr__(self) -> str:
        return f"{self.name} {self.extra_repr()} Backend"
    __str__ = __repr__

    def extra_repr(self):
        return self.version


class PyTorchBackend(BackendModule):
    alias = {"th", "torch", "pytorch"}

    def __init__(self):
        super().__init__()
        self.acceptable_names = self.acceptable_names.union({"pth", "th", "torch", "pytorch"})

    @property
    def version(self) -> str:
        return torch.__version__

    @property
    def name(self) -> str:
        return "PyTorch"

    @property
    def abbr(self) -> str:
        return "pytorch"


class TensorFlowBackend(BackendModule):
    alias = {"tf", "tensorflow"}

    def __init__(self):
        super().__init__()
        self.acceptable_names = self.acceptable_names.union({"tf", "tensorflow"})

    @property
    def version(self) -> str:
        return tensorflow.__version__

    @property
    def name(self) -> str:
        return "TensorFlow"

    @property
    def abbr(self) -> str:
        return "tensorflow"


class PyGBackend(PyTorchBackend):
    alias = {"pyg"}

    def __init__(self):
        super().__init__()
        self.acceptable_names = self.acceptable_names.union({"pyg"})

    @property
    def version(self) -> str:
        import torch_geometric
        return torch_geometric.__version__

    @property
    def name(self) -> str:
        return "PyTorch Geometric"

    @property
    def abbr(self) -> str:
        return "pyg"

    def extra_repr(self):
        return f"{super().extra_repr()} (PyTorch {torch.__version__})"


class DGLTensorFlowBackend(TensorFlowBackend):

    alias = {"dgl_tf"}

    def __init__(self):
        super().__init__()
        self.acceptable_names = self.acceptable_names.union({"dgl_tf"})

    @property
    def version(self) -> str:
        import dgl
        return dgl.__version__

    @property
    def name(self) -> str:
        return "DGL TensorFlow"

    @property
    def abbr(self) -> str:
        return "dgl_tf"

    def extra_repr(self):
        return f"{super().extra_repr()} (TensorFlow {tensorflow.__version__})"


class DGLPyTorchBackend(PyTorchBackend):
    alias = {"dgl_torch", "dgl_th", "dgl"}

    def __init__(self):
        super().__init__()
        self.acceptable_names = self.acceptable_names.union({"dgl_torch", "dgl_th", "dgl"})

    @property
    def version(self) -> str:
        import dgl
        return dgl.__version__

    @property
    def name(self) -> str:
        return "DGL PyTorch"

    @property
    def abbr(self) -> str:
        return "dgl_torch"

    def extra_repr(self):
        return f"{super().extra_repr()} (PyTorch {torch.__version__})"
