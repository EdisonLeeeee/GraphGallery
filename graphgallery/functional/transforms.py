__all__ = ["Transform", "NullTransform",
           "SparseTransform", "DenseTransform",
           "EdgeTransform", "GraphTransform",
           "TensorTransform"]


class Transform:

    def __init__(self):
        super().__init__()

    def __call__(self):
        raise NotImplementedError

    def extra_repr(self):
        return ""

    def __repr__(self):
        return f"{self.__class__.__name__}({self.extra_repr()})"
    __str__ = __repr__


class NullTransform(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, inputs):
        return inputs


class SparseTransform(Transform):
    pass


class DenseTransform(Transform):
    pass


class GraphTransform(Transform):
    pass


class EdgeTransform(Transform):
    pass


class TensorTransform(Transform):
    pass
