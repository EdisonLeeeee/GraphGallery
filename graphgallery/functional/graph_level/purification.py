import warnings
import numpy as np
import scipy.sparse as sp

from ..transform import GraphTransform
from ..transform import Transform
from ..edge_level import asedge
from ..sparse import remove_edge
import graphgallery as gg

__all__ = ["JaccardPurification", "CosinePurification", "SVD", "jaccard_similarity",
           "jaccard_purification", "cosine_purification", "svd", "cosine_similarity"]


_epsilon = 1e-7


def singleton_filter(edge, adj_matrix):
    """
    Filter edges that, if removed, would turn one or more nodes 
    into singleton nodes.

    Parameters
    ----------
    edge: np.array, shape [M, 2] or [2. M], where M is the number of input edges.
    adj_matrix: sp.sparse_matrix, shape [num_nodes, num_nodes]
        The input adjacency matrix.

    Returns
    -------
    np.array, shape [M, 2], 
        the edges that removed will not generate singleton nodes.
    """

    edge = asedge(edge, shape="row_wise")  # shape [M, 2]
    if edge.size == 0:
        warnings.warn("No edges found.", UserWarning)
        return edge

    degs = adj_matrix.sum(1).A1
    existing_edge = adj_matrix.tocsr(copy=False)[edge[:, 0], edge[:, 1]].A1

    if existing_edge.size > 0:
        edge_degrees = degs[edge] - 2 * existing_edge[:, None] + 1
    else:
        edge_degrees = degs[edge] + 1

    mask = np.logical_and(edge_degrees[:, 0] > 0, edge_degrees[:, 1] > 0)
    return edge[mask]


def jaccard_similarity(A, B):
    intersection = np.count_nonzero(A * B, axis=1)
    J = intersection * 1.0 / (np.count_nonzero(A, axis=1) + np.count_nonzero(B, axis=1) + intersection + _epsilon)
    return J


def cosine_similarity(A, B):
    inner_product = (A * B).sum(1)
    C = inner_product / (np.sqrt(np.square(A).sum(1)) * np.sqrt(np.square(B).sum(1)) + _epsilon)
    return C

# Using PyTorch
# def cosine_similarity(A, B):
#     inner_product = (A * B).sum(1)
#     C = inner_product / (torch.norm(A, 2, 1) * torch.norm(B, 2, 1) + 1e-7)
#     return C


def filter_edges_by_similarity(adj_matrix, attr_matrix,
                               similarity_fn, threshold=0.01,
                               allow_singleton=False):

    rows, cols = adj_matrix.nonzero()

    A = attr_matrix[rows]
    B = attr_matrix[cols]
    S = similarity_fn(A, B)
    idx = np.where(S <= threshold)[0]
    flips = np.vstack([rows[idx], cols[idx]])
    if not allow_singleton and flips.size > 0:
        flips = singleton_filter(flips, adj_matrix)
    return flips


def jaccard_purification(adj_matrix, attr_matrix, threshold=0.01, allow_singleton=False):
    return filter_edges_by_similarity(adj_matrix, attr_matrix,
                                      similarity_fn=jaccard_similarity,
                                      threshold=threshold,
                                      allow_singleton=allow_singleton)


def cosine_purification(adj_matrix, attr_matrix, threshold=0.01, allow_singleton=False):
    return filter_edges_by_similarity(adj_matrix, attr_matrix,
                                      similarity_fn=cosine_similarity,
                                      threshold=threshold,
                                      allow_singleton=allow_singleton)


@Transform.register()
class JaccardPurification(GraphTransform):

    def __init__(self, threshold=0., allow_singleton=False):
        super().__init__()
        self.threshold = threshold
        self.allow_singleton = allow_singleton
        self.flips = None

    def __call__(self, graph):
        assert isinstance(graph, gg.data.HomoGraph), type(graph)
        # TODO, multiple graph
        assert not graph.is_multiple(), "NOT Supported for multiple graph"
        graph = graph.copy()
        adj_matrix = graph.adj_matrix
        attr_matrix = graph.attr_matrix
        structure_flips = jaccard_purification(adj_matrix, attr_matrix,
                                               threshold=self.threshold,
                                               allow_singleton=self.allow_singleton)
        self.flips = structure_flips
        graph.update(adj_matrix=remove_edge(adj_matrix, structure_flips, symmetric=False))
        return graph

    def extra_repr(self):
        return f"threshold={self.threshold}, allow_singleton={self.allow_singleton}"


@Transform.register()
class CosinePurification(GraphTransform):

    def __init__(self, threshold=0., allow_singleton=False):
        super().__init__()
        self.threshold = threshold
        self.allow_singleton = allow_singleton
        self.flips = None

    def __call__(self, graph):
        assert isinstance(graph, gg.data.HomoGraph), type(graph)
        # TODO: multiple graph
        assert not graph.is_multiple(), "NOT Supported for multiple graph"
        graph = graph.copy()
        adj_matrix = graph.adj_matrix
        attr_matrix = graph.attr_matrix
        structure_flips = cosine_purification(adj_matrix, attr_matrix,
                                              threshold=self.threshold,
                                              allow_singleton=self.allow_singleton)

        self.flips = structure_flips
        graph.update(adj_matrix=remove_edge(adj_matrix, structure_flips, symmetric=False))
        return graph

    def extra_repr(self):
        return f"threshold={self.threshold}, allow_singleton={self.allow_singleton}"


@Transform.register()
class SVD(GraphTransform):

    def __init__(self, k=50, threshold=0.01, binaryzation=False):
        super().__init__()
        self.collect(locals())

    def __call__(self, graph):
        assert isinstance(graph, gg.data.HomoGraph), type(graph)
        # TODO: multiple graph
        assert not graph.is_multiple(), "NOT Supported for multiple graph"
        graph = graph.copy()
        adj_matrix = svd(graph.adj_matrix, k=self.k,
                         threshold=self.threshold,
                         binaryzation=self.binaryzation)
        graph.update(adj_matrix=adj_matrix)
        return graph


def svd(adj_matrix, k=50, threshold=0.01, binaryzation=False):
    adj_matrix = adj_matrix.asfptype()

    U, S, V = sp.linalg.svds(adj_matrix, k=k)
    adj_matrix = (U * S) @ V

    if threshold is not None:
        # sparsification
        adj_matrix[adj_matrix <= threshold] = 0.

    adj_matrix = sp.csr_matrix(adj_matrix)

    if binaryzation:
        # TODO
        adj_matrix.data[adj_matrix.data > 0] = 1.0

    return adj_matrix
