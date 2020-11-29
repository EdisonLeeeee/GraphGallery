from typing import Union, List, Tuple, Any, Callable, Optional

import graphgallery as gg
from graphgallery.functional import GDC
from graphgallery.functional import SVD
from graphgallery.functional import NormalizeAdj
from graphgallery.functional import AddSelfLoops
from graphgallery.functional import NormalizeAttr
from graphgallery.functional import WaveletBasis
from graphgallery.functional import ChebyBasis
from graphgallery.functional import NeighborSampler
from graphgallery.functional import GraphPartition
from graphgallery.functional import SparseAdjToEdge
from graphgallery.functional import SparseReshape

from .transforms import *
from .functions import *

__all__ = ['get', 'Compose']

_TRANSFORMS = {"gdc": GDC,
               "svd": SVD,
               "normalize_adj": NormalizeAdj,
               "normalize_attr": NormalizeAttr,
               "add_selfloops": AddSelfLoops,
               "wavelet_basis": WaveletBasis,
               "cheby_basis": ChebyBasis,
               "neighbor_sampler": NeighborSampler,
               "graph_partition": GraphPartition,
               "sparse_adj_to_edge": SparseAdjToEdge,
               "sparse_reshape": SparseReshape}

_ALLOWED = set(list(_TRANSFORMS.keys()))


class Compose(Transform):
    def __init__(self, *transforms: Union[str, Transform, None, List, Tuple, "Compose"],
                 **kwargs):
        self.transforms = [get(transform) for transform in transforms]

    def __call__(self, inputs: Any):
        for transform in self.transforms:
            if isinstance(inputs, tuple):
                inputs = transform(*inputs)
            else:
                inputs = transform(inputs)

        return inputs

    def add(self, transform: Union[str, Transform, None, List, Tuple, "Compose"]):
        self.transforms.append(get(transform))

    def pop(self, index: int = -1) -> Transform:
        """Remove and return 'transforms' at index (default last)."""
        return self.transforms.pop(index=-1)

    def extra_repr(self):
        format_string = ""
        for t in self.transforms:
            format_string += f'\n    {t}'
        return format_string


def get(transform: Union[str, Transform, None, List, Tuple, "Compose"]) -> Transform:
    if gg.is_listlike(transform):
        return Compose(*transform)

    if isinstance(transform, Transform) or callable(transform):
        return transform
    elif transform is None:
        return NullTransform()
    _transform = str(transform).lower()
    _transform = _TRANSFORMS.get(_transform, None)
    if _transform is None:
        raise ValueError(
            f"Unknown transform: '{transform}', expected a string, callable function or None.")
    return _transform()
