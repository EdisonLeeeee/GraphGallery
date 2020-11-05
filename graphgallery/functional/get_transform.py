from typing import Union, List, Tuple

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
# from graphgallery.functional import EdgeToSparseAdj
from graphgallery.functional import SparseReshape

from .transforms import *
from .ops import *

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
#                 "edge_to_sparse_adj": EdgeToSparseAdj,
                "sparse_reshape": SparseReshape}

_ALLOWED = set(list(_TRANSFORMS.keys()))


class Compose(Transform):
    def __init__(self, *transforms, **kwargs):
        """
        Initialize the transform

        Args:
            self: (todo): write your description
            transforms: (str): write your description
        """
        self.transforms = [get(transform) for transform in transforms]

    def __call__(self, inputs):
        """
        Return a copy of the module.

        Args:
            self: (todo): write your description
            inputs: (dict): write your description
        """
        for transform in self.transforms:
            if isinstance(inputs, tuple):
                inputs = transform(*inputs)
            else:
                inputs = transform(inputs)

        return inputs
    
    def add(self, transform):
        """
        Add a transformation to the given transformation.

        Args:
            self: (todo): write your description
            transform: (todo): write your description
        """
        self.transforms.append(get(transform))

    def __repr__(self):
        """
        Return a human - readable representation of this object.

        Args:
            self: (todo): write your description
        """
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


def get(transform: Union[str, Transform, None, List, Tuple, Compose]) -> Transform:
    """
    Get a transformation from the given transformation.

    Args:
        transform: (bool): write your description
    """
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
            f"Unknown transform: '{transform}', expected one of {_ALLOWED}, None or a callable function.")
    return _transform()
