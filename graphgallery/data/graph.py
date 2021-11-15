import numpy as np
import scipy.sparse as sp
from graphgallery import functional as gf

from .homograph import HomoGraph


class Graph(HomoGraph):
    """Attributed labeled homogeneous graph stored in sparse matrix form."""
    multiple = False

    @property
    def edge_index(self):
        edge_index, edge_weight = gf.sparse_adj_to_edge(self.adj_matrix)
        return edge_index

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
        G = self.copy()
        G.adj_matrix = gf.to_undirected(G.adj_matrix)
        return G

    def to_directed(self):
        """Convert to a directed graph."""
        G = self.copy()
        if self.is_directed():
            return G
        else:
            G.adj_matrix = gf.to_directed(G.adj_matrix)
        return G

    def to_unweighted(self):
        """Convert to an unweighted graph (set all edge weights to 1)."""
        G = self.copy()
        G.adj_matrix = gf.to_unweighted(G.adj_matrix)
        return G

    def eliminate_classes(self, threshold=0):
        """Remove nodes from graph that correspond to a class of which there are less
        or equal than 'threshold'. Those classes would otherwise break the training procedure.
        """
        if self.label is None:
            return self
        label = self.label
        counts = np.bincount(label)
        nodes_to_remove = []
        removed = 0
        left = []
        for _class, count in enumerate(counts):
            if count <= threshold:
                nodes_to_remove.extend(np.where(label == _class)[0])
                removed += 1
            else:
                left.append(_class)

        if removed > 0:
            # TODO: considering about metadata['class_names']
            G = self.subgraph(nodes_to_remove=nodes_to_remove)
            mapping = dict(zip(left, range(self.num_classes - removed)))
            G.label = np.asarray(list(
                map(lambda key: mapping[key], G.label)),
                dtype=np.int32)
            return G
        else:
            return self

    def eliminate_singleton(self):
        G = self.remove_self_loop()
        A = G.adj_matrix
        mask = np.logical_and(A.sum(0) == 0, A.sum(1) == 1)
        nodes_to_keep = mask.nonzero()[0]
        return G.subgraph(nodes_to_keep=nodes_to_keep)

    def from_flips(self, **flips):
        """Return a new graph from:
        'edge_flips' or 'feat_flips'
        """
        allowed = ("edge_flips", "feat_flips")
        g = self.copy()
        for k, v in flips.items():
            if v is None:
                continue
            if k == "edge_flips":
                g.update(adj_matrix=gf.flip_adj(g.adj_matrix, v))
            elif k == "feat_flips":
                g.update(attr_matrix=gf.flip_attr(g.attr_matrix, v))
            else:
                raise ValueError(f"Unrecognized key {k}, allowed: {allowed}.")
        return g

    def standardize(self):
        """Select the largest connected components (LCC) of 
        the unweighted/undirected/no-self-loop graph."""
        return gf.Standardize()(self)

    def to_networkx(self, directed: bool = True):
        """Get the network graph from adj_matrix."""
        return gf.to_networkx(self.adj_matrix,
                              directed=directed)

    def subgraph(self, *, nodes_to_keep=None, nodes_to_remove=None):
        return gf.subgraph(self, nodes_to_keep=nodes_to_keep,
                           nodes_to_remove=nodes_to_remove)

    def erase_node_attr(self, nodes, missing_rate=0.1):
        """PairNorm: Tackling Oversmoothing in GNNs
        <https://openreview.net/forum?id=rkecl1rtwB>
        ICLR 2020"""
        G = self.copy()
        G.attr_matrix = gf.erase_node_attr(G.attr_matrix,
                                           nodes=nodes,
                                           missing_rate=missing_rate)
        return G

    def erase_node_attr_except(self, nodes, missing_rate=0.1):
        """PairNorm: Tackling Oversmoothing in GNNs
        <https://openreview.net/forum?id=rkecl1rtwB>
        ICLR 2020"""
        G = self.copy()
        G.attr_matrix = gf.erase_node_attr_except(G.attr_matrix,
                                                  nodes=nodes,
                                                  missing_rate=missing_rate)
        return G
