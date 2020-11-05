import torch
import tensorflow
import numpy


__all__ = ['BackendModule', 'TensorFlowBackend',
           'PyTorchBackend', 'PyGBackend', 'DGLPyTorchBackend', 'DGLTensorFlowBackend']


class BackendModule:
    """Base Backend Module Class."""

    alias = {}

    def __init__(self):
        """
        Initialize the alias.

        Args:
            self: (todo): write your description
        """
        self.acceptable_names = self.alias

    @property
    def version(self) -> str:
        """
        Returns the version string.

        Args:
            self: (todo): write your description
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """
        Returns the name of the message.

        Args:
            self: (todo): write your description
        """
        raise NotImplementedError

    @property
    def abbr(self) -> str:
        """
        Returns the abbreviation.

        Args:
            self: (todo): write your description
        """
        return NotImplementedError

    def __eq__(self, value) -> bool:
        """
        Return true if the value.

        Args:
            self: (todo): write your description
            value: (todo): write your description
        """
        return str(value).lower() in self.acceptable_names

    def __str__(self) -> str:
        """
        Returns the string representation of the string.

        Args:
            self: (todo): write your description
        """
        return f"{self.name} {self.extra_repr()} Backend"

    def __repr__(self) -> str:
        """
        Return a repr repr of - friendly representation.

        Args:
            self: (todo): write your description
        """
        return self.__str__()

    def extra_repr(self):
        """
        Return the __repr__.

        Args:
            self: (todo): write your description
        """
        return self.version


class PyTorchBackend(BackendModule):
    alias = {"th", "torch", "pytorch"}

    def __init__(self):
        """
        Initialize the union

        Args:
            self: (todo): write your description
        """
        super().__init__()
        self.acceptable_names = self.acceptable_names.union({"th", "torch", "pytorch"})

    @property
    def version(self) -> str:
        """
        Return the version of the underlying version.

        Args:
            self: (todo): write your description
        """
        return torch.__version__

    @property
    def name(self) -> str:
        """
        Returns the name.

        Args:
            self: (todo): write your description
        """
        return "PyTorch"

    @property
    def abbr(self) -> str:
        """
        Abbrace string.

        Args:
            self: (todo): write your description
        """
        return "pytorch"


class TensorFlowBackend(BackendModule):
    alias = {"tf", "tensorflow"}

    def __init__(self):
        """
        Initialize the union

        Args:
            self: (todo): write your description
        """
        super().__init__()
        self.acceptable_names = self.acceptable_names.union({"tf", "tensorflow"})

    @property
    def version(self) -> str:
        """
        Returns the version of the sensor.

        Args:
            self: (todo): write your description
        """
        return tensorflow.__version__

    @property
    def name(self) -> str:
        """
        Returns the name.

        Args:
            self: (todo): write your description
        """
        return "TensorFlow"

    @property
    def abbr(self) -> str:
        """
        Abbrace string.

        Args:
            self: (todo): write your description
        """
        return "tensorflow"


class PyGBackend(PyTorchBackend):
    alias = {"pyg"}

    def __init__(self):
        """
        Initialize the union

        Args:
            self: (todo): write your description
        """
        super().__init__()
        self.acceptable_names = self.acceptable_names.union({"pyg"})

    @property
    def version(self) -> str:
        """
        Return the housech_gech

        Args:
            self: (todo): write your description
        """
        import torch_geometric
        return torch_geometric.__version__

    @property
    def name(self) -> str:
        """
        Returns the name.

        Args:
            self: (todo): write your description
        """
        return "PyTorch Geometric"

    @property
    def abbr(self) -> str:
        """
        Abbrace string.

        Args:
            self: (todo): write your description
        """
        return "pyg"

    def extra_repr(self):
        """
        Return a string representation of this object.

        Args:
            self: (todo): write your description
        """
        return f"{super().extra_repr()} (PyTorch {torch.__version__})"


class DGLTensorFlowBackend(TensorFlowBackend):

    alias = {"dgl_tf"}

    def __init__(self):
        """
        Initialize the union

        Args:
            self: (todo): write your description
        """
        super().__init__()
        self.acceptable_names = self.acceptable_names.union({"dgl_tf"})

    @property
    def version(self) -> str:
        """
        Return the version string.

        Args:
            self: (todo): write your description
        """
        import dgl
        return dgl.__version__

    @property
    def name(self) -> str:
        """
        Returns the name.

        Args:
            self: (todo): write your description
        """
        return "DGL TensorFlow"

    @property
    def abbr(self) -> str:
        """
        Abbrace string.

        Args:
            self: (todo): write your description
        """
        return "dgl_tf"

    def extra_repr(self):
        """
        Return a string representation of this object.

        Args:
            self: (todo): write your description
        """
        return f"{super().extra_repr()} (TensorFlow {tensorflow.__version__})"


class DGLPyTorchBackend(PyTorchBackend):
    alias = {"dgl_torch", "dgl_th", "dgl"}

    def __init__(self):
        """
        Initialize the union

        Args:
            self: (todo): write your description
        """
        super().__init__()
        self.acceptable_names = self.acceptable_names.union({"dgl_torch", "dgl_th", "dgl"})

    @property
    def version(self) -> str:
        """
        Return the version string.

        Args:
            self: (todo): write your description
        """
        import dgl
        return dgl.__version__

    @property
    def name(self) -> str:
        """
        Returns the name.

        Args:
            self: (todo): write your description
        """
        return "DGL PyTorch"

    @property
    def abbr(self) -> str:
        """
        Abbrace string.

        Args:
            self: (todo): write your description
        """
        return "dgl_torch"

    def extra_repr(self):
        """
        Return a string representation of this object.

        Args:
            self: (todo): write your description
        """
        return f"{super().extra_repr()} (PyTorch {torch.__version__})"
