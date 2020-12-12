import warnings
import numpy as np
import scipy.sparse as sp


from ..transforms import Transform
from ..edge_level import filter_singletons
import graphgallery as gg

__all__ = ["JaccardDetection", "CosineDetection",
           "jaccard_detection", "cosine_detection"]


def jaccard_similarity(A, B):
    intersection = np.count_nonzero(A * B, axis=1)
    J = intersection * 1.0 / (np.count_nonzero(A, axis=1) + np.count_nonzero(B, axis=1) + intersection + epsilon())
    return J


def cosine_similarity(A, B):
    inner_product = (A * B).sum(1)
    C = inner_product / (np.sqrt(np.square(A).sum(1)) * np.sqrt(np.square(B).sum(1)) + epsilon())
    return C


def filter_edges_by_similarity(adj_matrix, x,
                               similarity_fn, threshold=0.01,
                               allow_singleton=False):

    rows, cols = adj_matrix.nonzero()

    A = x[rows]
    B = x[cols]
    S = similarity_fn(A, B)
    idx = np.where(S <= threshold)[0]
    flips = np.vstack([rows[idx], cols[idx]])
    if not allow_singleton and flips.size > 0:
        flips = filter_singletons(flips, adj_matrix)
    return flips


def jaccard_detection(adj_matrix, x, threshold=0.01, allow_singleton=False):
    return filter_edges_by_similarity(adj_matrix, x, similarity_fn=jaccard_similarity,
                                      threshold=threshold,
                                      allow_singleton=allow_singleton)


def cosine_detection(adj_matrix, x, threshold=0.01, allow_singleton=False):
    return filter_edges_by_similarity(adj_matrix, x, similarity_fn=cosine_similarity,
                                      threshold=threshold,
                                      allow_singleton=allow_singleton)


class JaccardDetection(Transform):

    def __init__(self, threshold=0.01, allow_singleton=False):
        super().__init__()
        self.threshold = threshold
        self.allow_singleton = allow_singleton

    def __call__(self, graph):
        assert isinstance(graph, gg.data.HomoGraph), type(graph)
        # TODO, multiple graph
        assert not graph.multiple
        graph = graph.copy()
        adj_matrix = graph.adj_matrix
        node_attr = graph.node_attr
        structure_flips = jaccard_detection(adj_matrix, node_attr,
                                            threshold=self.threshold,
                                            allow_singleton=self.allow_singleton)

    def extra_repr(self):
        return f"threshold={self.threshold}, allow_singleton={self.allow_singleton}"


class CosineDetection(Transform):

    def __init__(self, threshold=0.01, allow_singleton=False):
        super().__init__()
        self.threshold = threshold
        self.allow_singleton = allow_singleton

    def __call__(self, graph):
        assert isinstance(graph, gg.data.HomoGraph), type(graph)
        # TODO: multiple graph
        assert not graph.multiple
        graph = graph.copy()
        adj_matrix = graph.adj_matrix
        node_attr = graph.node_attr
        structure_flips = cosine_detection(adj_matrix, node_attr,
                                           threshold=self.threshold,
                                           allow_singleton=self.allow_singleton)

    def extra_repr(self):
        return f"threshold={self.threshold}, allow_singleton={self.allow_singleton}"
