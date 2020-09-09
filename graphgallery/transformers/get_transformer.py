from typing import Union
from graphgallery.transformers import Transformer, NullTransformer
from graphgallery.transformers import NormalizeAdj
from graphgallery.transformers import AddSelfLoops
from graphgallery.transformers import NormalizeAttr
from graphgallery.transformers import WaveletBasis
from graphgallery.transformers import ChebyBasis
from graphgallery.transformers import NeighborSampler
from graphgallery.transformers import GraphPartition
from graphgallery.transformers import GDC
from graphgallery.transformers import SVD


_TRANSFORMER = {"normalize_adj": NormalizeAdj,
                "normalize_attr": NormalizeAttr,
                "add_selfloops": AddSelfLoops,
                "wavelet_basis": WaveletBasis,
                "cheby_basis": ChebyBasis,
                "neighbor_sampler": NeighborSampler,
                "graph_partition": GraphPartition,
                "gdc": GDC,
                "svd": SVD}

_ALLOWED = set(list(_TRANSFORMER.keys()))


def get(transformer: Union[str, Transformer, None]) -> Transformer:
    if isinstance(transformer, Transformer):
        return transformer
    elif transformer is None:
        return NullTransformer()
    _transformer = str(transformer).lower()
    _transformer = _TRANSFORMER.get(_transformer, None)
    if _transformer is None:
        raise ValueError(
            f"Unknown transformer: '{transformer}', expected one of {_ALLOWED} or None.")
    return _transformer()
