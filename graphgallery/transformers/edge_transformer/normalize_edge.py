import numpy as np
from graphgallery import floatx
from graphgallery.transformers import edge_transpose, add_selfloops_edge


def normalize_edge(edge_index, edge_weight=None, rate=-0.5, fill_weight=1.0):
    edge_index = edge_transpose(edge_index)
    
    n_nodes = edge_index.max() + 1
    
    if edge_weight is None:
        edge_weight = np.ones(edge_index.shape[1], dtype=floatx())
        
    if fill_weight:
        edge_index, edge_weight = add_selfloops_edge(
            edge_index, edge_weight, n_nodes=n_nodes, fill_weight=fill_weight)    
        
    degree = np.bincount(edge_index[0], weights=edge_weight)
    degree_power = np.power(degree, rate)
    row, col = edge_index
    edge_weight_norm = degree_power[row] * edge_weight * degree_power[col]
    
    return edge_index, edge_weight_norm

        