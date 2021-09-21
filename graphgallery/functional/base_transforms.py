__all__ = ["BaseTransform", "NullTransform",
           "SparseTransform", "DenseTransform",
           "EdgeTransform", "GraphTransform",
           "TensorTransform"]


class BaseTransform:

    def __init__(self):
        super().__init__()

    def __call__(self):
        raise NotImplementedError

    def collect(self, kwargs):
        assert isinstance(kwargs, dict)
        kwargs.pop('self', None)
        for k, v in kwargs.items():
            if not k.startswith("__"):
                setattr(self, k, v)

    def extra_repr(self):
        paras = self.__dict__
        if not paras:
            return ""
        formatstring = ""
        for k in sorted(paras):
            v = paras[k]
            formatstring += f"{k}={v}, "
        # escape `,` and space
        formatstring = formatstring[:-2]
        return formatstring

    def __repr__(self):
        return f"{self.__class__.__name__}({self.extra_repr()})"
    __str__ = __repr__


class NullTransform(BaseTransform):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, *inputs):
        if len(inputs) == 1:
            inputs, = inputs
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
