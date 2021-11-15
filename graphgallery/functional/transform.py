import graphgallery as gg
from .registry import Registry


__all__ = ["BaseTransform", "NullTransform",
           "SparseTransform", "DenseTransform",
           "EdgeTransform", "GraphTransform",
           "TensorTransform", "get", "Compose", "Transform"]

Transform = Registry("Transform")


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


class Compose(BaseTransform):
    def __init__(self, *base_transforms, **kwargs):
        self.base_transforms = [get(transform) for transform in base_transforms]

    def __call__(self, inputs):
        for transform in self.base_transforms:
            if isinstance(inputs, tuple):
                inputs = transform(*inputs)
            else:
                inputs = transform(inputs)

        return inputs

    def add(self, transform):
        self.base_transforms.append(get(transform))

    def pop(self, index: int = -1):
        """Remove and return 'base_transforms' at index (default last)."""
        return self.base_transforms.pop(index=-1)

    def extra_repr(self):
        format_string = ""
        for t in self.base_transforms:
            format_string += f'\n  {t},'
        if format_string:
            # replace last ``,`` as ``\n``
            format_string = format_string[:-1] + '\n'
        return format_string


def is_name_dict_tuple(transform):
    """Check the transform is somehow like
        ('normalize_adj', dict(rate=1.0))
    """

    return len(transform) == 2 and isinstance(transform[0], str) and isinstance(transform[1], dict)


def get(transform):
    if gg.is_listlike(transform) and not is_name_dict_tuple(transform):
        return Compose(*transform)

    if isinstance(transform, BaseTransform) or callable(transform):
        return transform
    elif transform is None:
        return NullTransform()

    transform_para = {}
    if isinstance(transform, tuple):
        transform, transform_para = transform
    original_transform = transform
    assert isinstance(transform, str), transform
    if transform not in Transform:
        transform = "".join(map(lambda s: s.title(), transform.split("_")))
        if transform not in Transform:
            raise ValueError(f"transform not found `{original_transform}`.")
    return Transform.get(transform)(**transform_para)
