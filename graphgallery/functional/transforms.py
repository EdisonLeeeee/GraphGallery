__all__ = ["BaseTransform", "NullTransform",
           "SparseTransform", "DenseTransform",
           "EdgeTransform", "GraphTransform",
           "TensorTransform"]


class BaseTransform:

    def __init__(self):
        super().__init__()

    def __call__(self):
        raise NotImplementedError

    def extra_repr(self):
        return ""

    def __repr__(self):
        return f"{self.__class__.__name__}({self.extra_repr()})"
    __str__ = __repr__


class NullTransform(BaseTransform):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, inputs):
        return inputs


class SparseTransform(BaseTransform):
    pass


class DenseTransform(BaseTransform):
    pass


class GraphTransform(BaseTransform):
    pass


class EdgeTransform(BaseTransform):
    pass


class TensorTransform(BaseTransform):
    pass
