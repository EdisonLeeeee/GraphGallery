import os
import numpy as np
import networkx as nx
import scipy.sparse as sp
from graphgallery import functional as gf

from typing import Union, Optional, List, Tuple, Any

from .homograph import HomoGraph
from .preprocess import create_subgraph


class Graph(HomoGraph):
    """Attributed labeled homogeneous graph stored in sparse matrix form."""
    multiple = False

    def to_EdgeGraph(self):
        raise NotImplementedError

    def neighbors(self, idx):
        """Get the indices of neighbors of a given node.

        Parameters
        ----------
        idx : int
            Index of the node whose neighbors are of interest.
        """
        return self.adj_matrix[idx].indices

    def to_undirected(self):
        """Convert to an undirected graph (make adjacency matrix symmetric)."""
        if self.is_weighted():
            raise RuntimeError(
                "Convert to unweighted graph first. Using 'graph.to_unweighted()'."
            )
        else:
            G = self.copy()
            A = G.adj_matrix
            A = A.maximum(A.T)
            G.adj_matrix = A
        return G

    def to_unweighted(self):
        """Convert to an unweighted graph (set all edge weights to 1)."""
        G = self.copy()
        A = G.adj_matrix
        G.adj_matrix = sp.csr_matrix(
            (np.ones_like(A.data), A.indices, A.indptr), shape=A.shape)
        return G

    def eliminate_selfloops(self):
        """Remove self-loops from the adjacency matrix."""
        G = self.copy()
        A = G.adj_matrix
        A = A - sp.diags(A.diagonal())
        A.eliminate_zeros()
        G.adj_matrix = A
        return G

    def eliminate_classes(self, threshold=0):
        """Remove nodes from graph that correspond to a class of which there are less
        or equal than 'threshold'. Those classes would otherwise break the training procedure.
        """
        if self.node_label is None:
            return self
        node_label = self.node_label
        counts = np.bincount(node_label)
        nodes_to_remove = []
        removed = 0
        left = []
        for _class, count in enumerate(counts):
            if count <= threshold:
                nodes_to_remove.extend(np.where(node_label == _class)[0])
                removed += 1
            else:
                left.append(_class)

        if removed > 0:
            # TODO: considering about metadata['class_names']
            G = self.subgraph(nodes_to_remove=nodes_to_remove)
            mapping = dict(zip(left, range(self.num_node_classes - removed)))
            G.node_label = np.asarray(list(
                map(lambda key: mapping[key], G.node_label)),
                dtype=np.int32)
            return G
        else:
            return self

    def eliminate_singleton(self):
        G = self.graph.eliminate_selfloops()
        A = G.adj_matrix
        mask = np.logical_and(A.sum(0) == 0, A.sum(1) == 1)
        nodes_to_keep = mask.nonzero()[0]
        return G.subgraph(nodes_to_keep=nodes_to_keep)

#     def add_selfloops(self, value=1.0):
#         """Set the diagonal."""
#         G = self.eliminate_selfloops()
#         A = G.adj_matrix
#         A = A + sp.diags(A.diagonal() + value)
#         A.eliminate_zeros()
#         G.adj_matrix = A
#         return G
    
    def from_flips(self, **flips):
        """Return a new graph from:
        'edge_flips' or 'nx_flips'
        """
        allowed = ("edge_flips", "nx_flips")
        g = self.copy()
        for k, v in flips.items():
            if v is None:
                continue
            if k=="edge_flips":
                g.update(adj_matrix=gf.flip_adj(g.adj_matrix, v))
            elif k=="nx_flips":
                g.update(node_attr=gf.flip_attr(g.node_attr, v))
            else:
                raise ValueError(f"Unrecognized key {k}, allowed: {allowed}.")
        return g
                

    def standardize(self):
        """Select the largest connected components (LCC) of 
        the unweighted/undirected/no-self-loop graph."""
        return gf.Standardize()(self)

    def nxgraph(self, directed: bool = True):
        """Get the network graph from adj_matrix."""
        if directed:
            create_using = nx.DiGraph
        else:
            create_using = nx.Graph
        return nx.from_scipy_sparse_matrix(self.adj_matrix,
                                           create_using=create_using)

    def subgraph(self, *, nodes_to_remove=None, nodes_to_keep=None):
        return create_subgraph(self,
                               nodes_to_remove=nodes_to_remove,
                               nodes_to_keep=nodes_to_keep)

    def erase_node_attr(self, nodes, missing_rate=0.1):
        """PairNorm: Tackling Oversmoothing in GNNs
        <https://openreview.net/forum?id=rkecl1rtwB>
        ICLR 2020"""
        assert 0 <= missing_rate <= 1
        if missing_rate > 0:
            node_erased = np.random.choice(nodes, size=int(len(nodes) * missing_rate), replace=False)
            self.node_attr[node_erased] = 0.

        return self

    def erase_node_attr_except(self, nodes, missing_rate=0.1):
        """PairNorm: Tackling Oversmoothing in GNNs
        <https://openreview.net/forum?id=rkecl1rtwB>
        ICLR 2020"""
        mask = gf.index_to_mask(nodes, self.num_nodes)
        erased = np.arange(self.num_nodes, dtype=nodes.dtype)
        return self.erase_node_attr(erased[~mask], missing_rate=missing_rate)
