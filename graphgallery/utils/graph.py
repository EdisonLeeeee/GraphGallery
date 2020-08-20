try:
    import metis
except ImportError:
    metis = None

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def metis_clustering(graph, n_clusters):
    """Partitioning graph using Metis"""
    _, parts = metis.part_graph(graph, n_clusters)
    return parts


def random_clustering(n_nodes, n_clusters):
    """Partitioning graph randomly"""
    partts = np.random.choice(n_clusters, size=n_nodes)
    return parts


def partition_graph(adj, x, graph, n_clusters, metis_partition=True):
    # partition graph
    if metis_partition:
        assert metis, "Please install `metis` package!"
        parts = metis_clustering(graph, n_clusters)
    else:
        parts = random_clustering(adj.shape[0], n_clusters)

    cluster_member = [[] for _ in range(n_clusters)]
    for node_index, part in enumerate(parts):
        cluster_member[part].append(node_index)

    mapper = {}

    batch_adj, batch_x = [], []

    for cluster in range(n_clusters):

        nodes = sorted(cluster_member[cluster])
        mapper.update({old_id: new_id for new_id, old_id in enumerate(nodes)})

        mini_adj = adj[nodes].tocsc()[:, nodes]
        mini_x = x[nodes]

        batch_adj.append(mini_adj)
        batch_x.append(mini_x)

    return batch_adj, batch_x, cluster_member


def construct_neighbors(adj, max_degree=25, self_loop=False):
    N = adj.shape[0]
    indices = adj.indices
    indptr = adj.indptr
    adj_dense = N * np.ones((N + 1, max_degree), dtype=np.int64)
    for nodeid in range(N):
        neighbors = indices[indptr[nodeid]:indptr[nodeid + 1]]

#         if not self_loop:
#             neighbors = np.setdiff1d(neighbors, [nodeid])
#         else:
#             neighbors = np.intersect1d(neighbors, [nodeid])

        size = neighbors.size
        if size == 0:
            continue

        if size > max_degree:
            neighbors = np.random.choice(neighbors, max_degree, replace=False)
        elif size < max_degree:
            neighbors = np.random.choice(neighbors, max_degree, replace=True)

        adj_dense[nodeid] = neighbors

    np.random.shuffle(adj_dense.T)
    return adj_dense


def sample_neighbors(adj, nodes, n_neighbors):
    np.random.shuffle(adj.T)
    return adj[nodes, :n_neighbors]


def get_indice_graph(adj, indices, size=np.inf, dropout=0.):
    if dropout > 0.:
        indices = np.random.choice(indices, int(indices.size*(1-dropout)), False)
    neighbors = adj[indices].sum(axis=0).nonzero()[0]
    if neighbors.size > size - indices.size:
        neighbors = np.random.choice(list(neighbors), size-len(indices), False)
    indices = np.union1d(indices, neighbors)
    return indices


def largest_connected_components(adj, n_components=1):
    """Select the largest connected components in the graph.

    Parameters
    ----------
    sparse_graph : scipy.sparse.csr_matrix
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.

    Returns
    -------
    sparse_graph : gust.SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.

    """
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep
    ]
    return nodes_to_keep