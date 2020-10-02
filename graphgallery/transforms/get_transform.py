from typing import Union, List, Tuple
from graphgallery.transforms import GDC
from graphgallery.transforms import SVD
from graphgallery.utils.type_check import is_list_like
from graphgallery.transforms import Transform, NullTransform
from graphgallery.transforms import NormalizeAdj
from graphgallery.transforms import AddSelfLoops
from graphgallery.transforms import NormalizeAttr
from graphgallery.transforms import WaveletBasis
from graphgallery.transforms import ChebyBasis
from graphgallery.transforms import NeighborSampler
from graphgallery.transforms import GraphPartition
from graphgallery.transforms import SparseAdjToSparseEdges
from graphgallery.transforms import SparseEdgesToSparseAdj
from graphgallery.transforms import SparseReshape




_TRANSFORMER = {"gdc": GDC,
                "svd": SVD,
                "normalize_adj": NormalizeAdj,
                "normalize_attr": NormalizeAttr,
                "add_selfloops": AddSelfLoops,
                "wavelet_basis": WaveletBasis,
                "cheby_basis": ChebyBasis,
                "neighbor_sampler": NeighborSampler,
                "graph_partition": GraphPartition,
                "sparse_adj_to_sparse_edges": SparseAdjToSparseEdges,
                "sparse_edges_to_sparse_adj": SparseEdgesToSparseAdj,
                "sparse_reshape": SparseReshape}

_ALLOWED = set(list(_TRANSFORMER.keys()))


class Compose(Transform):
    def __init__(self, *transforms, **kwargs):
        self.transforms = tuple(get(transform) for transform in transforms)
        
    def __call__(self, inputs):
        for transform in self.transforms:
            if isinstance(inputs, tuple):
                inputs = transform(*inputs)
            else:
                inputs = transform(inputs)
            
        return inputs
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string    
    
    
def get(transform: Union[str, Transform, None, List, Tuple, Compose]) -> Transform:
    if is_list_like(transform):
        return Compose(*transform)
    
    if isinstance(transform, Transform) or callable(transform):
        return transform
    elif transform is None:
        return NullTransform()
    _transform = str(transform).lower()
    _transform = _TRANSFORMER.get(_transform, None)
    if _transform is None:
        raise ValueError(
            f"Unknown transform: '{transform}', expected one of {_ALLOWED}, None or a callable function.")
    return _transform()

