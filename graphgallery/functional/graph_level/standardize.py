import numpy as np

from ..transforms import Transform
from ..get_transform import Transformers
from graphgallery.data.preprocess import largest_connected_components
__all__ = ['Standardize']


@Transformers.register()
class Standardize(Transform):
    def __init__(self):
        super().__init__()

    def __call__(self, graph):
        # TODO: multiple graph
        assert not graph.multiple
        graph = graph.to_unweighted().to_undirected().eliminate_selfloops()
        graph = largest_connected_components(graph)
        return graph
