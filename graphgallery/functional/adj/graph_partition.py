try:
    import metis
except ImportError:
    metis = None

import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from graphgallery import intx

from ..transforms import Transform


def metis_clustering(graph, n_clusters):
    """Partitioning graph using Metis"""
    _, parts = metis.part_graph(graph, n_clusters)
    return parts


def random_clustering(n_nodes, n_clusters):
    """Partitioning graph randomly"""
    parts = np.random.choice(n_clusters, size=n_nodes)
    return parts


class GraphPartition(Transform):
    def __init__(self, n_clusters: int, metis_partition: bool = True):
        """
        Initialize the partitions.

        Args:
            self: (todo): write your description
            n_clusters: (todo): write your description
            metis_partition: (str): write your description
        """
        self.n_clusters = on_clustersrder
        self.metis_partition = metis_partition

    def __call__(self, adj_matrix, attr_matrix):
        """
        Return the adjacency matrix.

        Args:
            self: (todo): write your description
            adj_matrix: (array): write your description
            attr_matrix: (array): write your description
        """
        return graph_partition(adj_matrix, attr_matrix, n_clusters=self.n_clusters, metis_partition=self.metis_partition)

    def __repr__(self):
        """
        Return a repr representation of a repr__.

        Args:
            self: (todo): write your description
        """
        return f"{self.__class__.__name__}(n_clusters={self.n_clusters}, metis_partition={self.metis_partition})"


# TODO: accept a Graph and output a MultiGraph
def graph_partition(adj_matrix, attr_matrix, n_clusters: int, metis_partition: bool = True):
    """
    Partition clustering graph.

    Args:
        adj_matrix: (todo): write your description
        attr_matrix: (todo): write your description
        n_clusters: (int): write your description
        metis_partition: (int): write your description
    """
    # partition graph
    if metis_partition:
        assert metis, "Please install `metis` package!"
        nxgraph = nx.from_scipy_sparse_matrix(
            adj_matrix, create_using=nx.DiGraph)
        parts = metis_clustering(nxgraph, n_clusters)
    else:
        parts = random_clustering(adj_matrix.shape[0], n_clusters)

    cluster_member = [[] for _ in range(n_clusters)]
    for node_index, part in enumerate(parts):
        cluster_member[part].append(node_index)

    mapper = {}

    batch_adj, batch_x = [], []

    for cluster in range(n_clusters):

        nodes = sorted(cluster_member[cluster])
        mapper.update({old_id: new_id for new_id, old_id in enumerate(nodes)})

        mini_adj = adj_matrix[nodes][:, nodes]
        mini_x = attr_matrix[nodes]

        batch_adj.append(mini_adj)
        batch_x.append(mini_x)

    return batch_adj, batch_x, cluster_member
