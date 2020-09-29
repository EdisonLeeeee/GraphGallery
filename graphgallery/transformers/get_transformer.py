from typing import Union, List, Tuple
from graphgallery.transformers import GDC
from graphgallery.transformers import SVD
from graphgallery.utils.type_check import is_list_like
from graphgallery.transformers import Transformer, NullTransformer
from graphgallery.transformers import NormalizeAdj
from graphgallery.transformers import AddSelfLoops
from graphgallery.transformers import NormalizeAttr
from graphgallery.transformers import WaveletBasis
from graphgallery.transformers import ChebyBasis
from graphgallery.transformers import NeighborSampler
from graphgallery.transformers import GraphPartition
from graphgallery.transformers import SparseAdjToSparseEdges
from graphgallery.transformers import SparseEdgesToSparseAdj
from graphgallery.transformers import SparseReshape




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


class Pipeline(Transformer):
    def __init__(self, *transformers, **kwargs):
        self.transformers = tuple(get(transformer) for transformer in transformers)
        
    def __call__(self, inputs):
        for transformer in self.transformers:
            if isinstance(inputs, tuple):
                inputs = transformer(*inputs)
            else:
                inputs = transformer(inputs)
            
        return inputs
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transformers:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string    
    
#     def __repr__(self):
#         return f"{self.__class__.__name__}(transformer={self.transformers})"
    
def get(transformer: Union[str, Transformer, None, List, Tuple, Pipeline]) -> Transformer:
    if is_list_like(transformer):
        return Pipeline(*transformer)
    
    if isinstance(transformer, Transformer) or callable(transformer):
        return transformer
    elif transformer is None:
        return NullTransformer()
    _transformer = str(transformer).lower()
    _transformer = _TRANSFORMER.get(_transformer, None)
    if _transformer is None:
        raise ValueError(
            f"Unknown transformer: '{transformer}', expected one of {_ALLOWED}, None or a callable function.")
    return _transformer()

