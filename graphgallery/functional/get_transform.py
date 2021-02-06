import graphgallery as gg
from .registry import Registry

from .transforms import *
from .functions import *

__all__ = ["get", "Compose", "Transformers"]


Transformers = Registry("Transformers")


class Compose(Transform):
    def __init__(self, *transforms,
                 **kwargs):
        self.transforms = [get(transform) for transform in transforms]

    def __call__(self, inputs):
        for transform in self.transforms:
            if isinstance(inputs, tuple):
                inputs = transform(*inputs)
            else:
                inputs = transform(inputs)

        return inputs

    def add(self, transform):
        self.transforms.append(get(transform))

    def pop(self, index: int = -1):
        """Remove and return 'transforms' at index (default last)."""
        return self.transforms.pop(index=-1)

    def extra_repr(self):
        format_string = ""
        for t in self.transforms:
            format_string += f'\n    {t}'
        return format_string


def is_name_dict_tuple(transform):
    return len(transform) == 2 and isinstance(transform[0], str) and isinstance(transform[1], dict)


def get(transform):
    if gg.is_listlike(transform) and not is_name_dict_tuple(transform):
        return Compose(*transform)

    if isinstance(transform, Transform) or callable(transform):
        return transform
    elif transform is None:
        return NullTransform()

    transform_para = {}
    if isinstance(transform, tuple):
        transform, transform_para = transform
    original_transform = transform
    assert isinstance(transform, str), transform
    if transform not in Transformers:
        transform = "".join(map(lambda s: s.title(), transform.split("_")))
        if transform not in Transformers:
            raise ValueError(f"transform not found `{original_transform}`.")
    return Transformers.get(transform)(**transform_para)
