try:
    import metis
except ImportError:
    metis = None

import numpy as np
import scipy.sparse as sp


def metis_clustering(graph, n_cluster):

    _, parts = metis.part_graph(graph, n_cluster)

    return parts


def partition_graph(adj, features, labels, graph, n_cluster):

    assert metis is not None, "Please install `metis` package!"    
    # partition graph
    parts = metis_clustering(graph, n_cluster)
    
    cluster_member = [[] for _ in range(n_cluster)]
    for node_index, part in enumerate(parts):
        cluster_member[part].append(node_index)

    mapper = {}
    
    batch_adj, batch_features, batch_labels = [], [], []
    
    for cluster in range(n_cluster):
        
        nodes = sorted(cluster_member[cluster])
        mapper.update({old_id: new_id for new_id, old_id in enumerate(nodes)})
        
        mini_adj = adj[nodes].tocsc()[:, nodes]
        mini_features = features[nodes]

        batch_adj.append(mini_adj)
        batch_features.append(mini_features)
        batch_labels.append(labels[nodes])
        
    return batch_adj, batch_features, batch_labels, cluster_member, mapper


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
    neighbors = adj[indices].sum(axis=0).nonzero()[1]
    if neighbors.size > size - indices.size:
        neighbors = np.random.choice(list(neighbors), size-len(indices), False)
    indices = np.union1d(indices, neighbors)
    return indices
