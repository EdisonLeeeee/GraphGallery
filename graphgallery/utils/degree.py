import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict

@njit
def mixing_matrix(edges, in_deg, out_deg, mapping):
    n_degrees = len(mapping)
    M = np.zeros((n_degrees, n_degrees))

    for i in range(edges.shape[0]):
        u = edges[i][0]
        v = edges[i][1]
        du = out_deg[u]
        dv = in_deg[v]
        x = mapping[du]
        y = mapping[dv]
        M[x, y] += 1
        M[y, x] += 1  
    return M

def degree_mixing_matrix(adj, normalize=True):
    out_deg = adj.sum(1).A1.astype('int64')
    in_deg = adj.sum(0).A1.astype('int64')
    deg_set = np.union1d(out_deg, in_deg)

    mapping = Dict.empty(
        key_type=types.int64,
        value_type=types.int64,
    )
    for x, y in enumerate(deg_set):
        mapping[y] = x

    edges = np.transpose(adj.nonzero()).astype('int64')
    
    M = mixing_matrix(edges, in_deg, out_deg, mapping)
    
    if normalize:
        M /= M.sum()
        
    return M, mapping
    
def numeric_ac(M, mapping):
    # M is a numpy matrix or array
    # numeric assortativity coefficient, pearsonr
    x = y = np.array(list(mapping.keys()), dtype='float')
    a = M.sum(0)
    b = M.sum(1)
    # D(x) = E(x^2) - [E(x)]^2
    vara = (a * x**2).sum() - ((a * x).sum())**2
    varb = (b * x**2).sum() - ((b * x).sum())**2
    xy = np.outer(x, y)
    ab = np.outer(a, b)

    return (xy * (M - ab)).sum() / vara

def degree_assortativity_coefficient(adj):
    M, mapping = degree_mixing_matrix(adj)
    return numeric_ac(M, mapping)

