import sys
import torch
import importlib

__all__ = ['BackendModule', 'TensorFlowBackend',
           'PyTorchBackend', 'PyGBackend',
           'DGLBackend']


class BackendModule:
    """Base Backend Module Class."""

    alias = set()

    def __init__(self, module=None):
        self.acceptable_names = self.alias

        if module is not None:
            try:
                self.module = importlib.import_module(module)
            except ImportError as e:
                print(f"Something went wrong when import `{module}`.", file=sys.stderr)
                raise e
        else:
            self.module = None

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

    def device(self, ctx):
        return self.module.device(ctx)


class PyTorchBackend(BackendModule):
    alias = {"th", "torch", "pytorch"}

    def __init__(self, module='torch'):
        super().__init__(module=module)
        self.acceptable_names = self.acceptable_names.union({"pth", "th", "torch", "pytorch"})

    @property
    def version(self):
        return self.module.__version__

    @property
    def name(self):
        return "PyTorch"

    @property
    def abbr(self):
        return "pytorch"


class TensorFlowBackend(BackendModule):
    alias = {"tf", "tensorflow"}

    def __init__(self, module='tensorflow'):
        super().__init__(module=module)
        self.acceptable_names = self.acceptable_names.union({"tf", "tensorflow"})

    @property
    def version(self):
        return self.module.__version__

    @property
    def name(self):
        return "TensorFlow"

    @property
    def abbr(self):
        return "tensorflow"


class PyGBackend(PyTorchBackend):
    alias = {"pyg"}

    def __init__(self):
        super().__init__(module='torch_geometric')
        self.acceptable_names = self.acceptable_names.union({"pyg"})

    @property
    def version(self):
        return self.module.__version__

    @property
    def name(self):
        return "PyTorch Geometric"

    @property
    def abbr(self):
        return "pyg"

    def extra_repr(self):
        return f"{super().extra_repr()} (PyTorch {torch.__version__})"

    def device(self, ctx):
        return torch.device(ctx)


class DGLBackend(PyTorchBackend):
    alias = {"dgl_torch", "dgl_th", "dgl"}

    def __init__(self):
        super().__init__(module='dgl')
        self.acceptable_names = self.acceptable_names.union({"dgl_torch", "dgl_th", "dgl"})

    @property
    def version(self):
        return self.module.__version__

    @property
    def name(self):
        return "DGL PyTorch"

    @property
    def abbr(self):
        return "dgl"

    def extra_repr(self):
        return f"{super().extra_repr()} (PyTorch {torch.__version__})"

    def device(self, ctx):
        return torch.device(ctx)
