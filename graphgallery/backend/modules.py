import torch
import tensorflow


__all__ = ['BackendModule', 'TensorFlowBackend',
           'PyTorchBackend', 'PyGBackend',
           'DGLPyTorchBackend', 'DGLTensorFlowBackend']


class BackendModule:
    """Base Backend Module Class."""

    alias = {}

    def __init__(self):
        self.acceptable_names = self.alias

    @property
    def version(self):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError

    @property
    def abbr(self):
        # used for specifying the module
        return NotImplementedError

    def __eq__(self, value) -> bool:
        return str(value).lower() in self.acceptable_names

    def __repr__(self):
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
    def version(self):
        return torch.__version__

    @property
    def name(self):
        return "PyTorch"

    @property
    def abbr(self):
        return "pytorch"


class TensorFlowBackend(BackendModule):
    alias = {"tf", "tensorflow"}

    def __init__(self):
        super().__init__()
        self.acceptable_names = self.acceptable_names.union({"tf", "tensorflow"})

    @property
    def version(self):
        return tensorflow.__version__

    @property
    def name(self):
        return "TensorFlow"

    @property
    def abbr(self):
        return "tensorflow"


class PyGBackend(PyTorchBackend):
    alias = {"pyg"}

    def __init__(self):
        super().__init__()
        self.acceptable_names = self.acceptable_names.union({"pyg"})

    @property
    def version(self):
        import torch_geometric
        return torch_geometric.__version__

    @property
    def name(self):
        return "PyTorch Geometric"

    @property
    def abbr(self):
        return "pyg"

    def extra_repr(self):
        return f"{super().extra_repr()} (PyTorch {torch.__version__})"


class DGLTensorFlowBackend(TensorFlowBackend):
    alias = {"dgl_tf"}

    def __init__(self):
        super().__init__()
        self.acceptable_names = self.acceptable_names.union({"dgl_tf"})

    @property
    def version(self):
        import dgl
        return dgl.__version__

    @property
    def name(self):
        return "DGL TensorFlow"

    @property
    def abbr(self):
        return "dgl_tf"

    def extra_repr(self):
        return f"{super().extra_repr()} (TensorFlow {tensorflow.__version__})"


class DGLPyTorchBackend(PyTorchBackend):
    alias = {"dgl_torch", "dgl_th", "dgl"}

    def __init__(self):
        super().__init__()
        self.acceptable_names = self.acceptable_names.union({"dgl_torch", "dgl_th", "dgl"})

    @property
    def version(self):
        import dgl
        return dgl.__version__

    @property
    def name(self):
        return "DGL PyTorch"

    @property
    def abbr(self):
        return "dgl_torch"

    def extra_repr(self):
        return f"{super().extra_repr()} (PyTorch {torch.__version__})"
