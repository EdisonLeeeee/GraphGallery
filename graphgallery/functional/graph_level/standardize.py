import numpy as np

from ..transforms import Transform
from graphgallery.data.preprocess import largest_connected_components
__all__ = ['Standardize']


class Standardize(Transform):
    def __init__(self):
        super().__init__()

    def __call__(self, graph):
        # TODO: multiple graph
        graph = graph.to_unweighted().to_undirected().eliminate_selfloops()
        graph = largest_connected_components(graph, 1)
        return graph
