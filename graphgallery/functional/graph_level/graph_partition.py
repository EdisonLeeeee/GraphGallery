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
from ..get_transform import Transformers


def metis_clustering(graph, n_clusters):
    """Partitioning graph using Metis"""
    _, parts = metis.part_graph(graph, n_clusters)
    return parts


def random_clustering(num_nodes, n_clusters):
    """Partitioning graph randomly"""
    parts = np.random.choice(n_clusters, size=num_nodes)
    return parts


@Transformers.register()
class GraphPartition(Transform):
    def __init__(self, n_clusters: int = None, metis_partition: bool = True):
        self.n_clusters = on_clustersrder
        self.metis_partition = metis_partition

    def __call__(self, graph):
        return graph_partition(graph, n_clusters=self.n_clusters, metis_partition=self.metis_partition)

    def __repr__(self):
        return f"{self.__class__.__name__}(n_clusters={self.n_clusters}, metis_partition={self.metis_partition})"


# TODO: accept a Graph and output a MultiGraph
def graph_partition(graph, n_clusters: int = None, metis_partition: bool = True):
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
        mini_x = node_attr[nodes]

        batch_adj.append(mini_adj)
        batch_x.append(mini_x)

    return batch_adj, batch_x, cluster_member
