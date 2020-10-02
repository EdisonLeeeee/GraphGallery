import numpy as np
from graphgallery import floatx
from graphgallery.transforms import edge_transpose


def add_selfloops_edge(edge_index, edge_weight, n_nodes=None, fill_weight=1.0):
    edge_index = edge_transpose(edge_index)
            
    if n_nodes is None:
        n_nodes = edge_index.max() + 1

    if edge_weight is None:
        edge_weight = np.ones(edge_index.shape[1], dtype=floatx())

    diagnal_edge_index = np.asarray(np.diag_indices(n_nodes)).astype(edge_index.dtype, copy=False)
    
    updated_edge_index = np.hstack([edge_index, diagnal_edge_index])

    diagnal_edge_weight = np.zeros(n_nodes, dtype=floatx()) + fill_weight
    updated_edge_weight = np.hstack([edge_weight, diagnal_edge_weight])

    return updated_edge_index, updated_edge_weight

        