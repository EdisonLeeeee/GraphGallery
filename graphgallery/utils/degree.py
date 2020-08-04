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
    """Compute degree assortativity of graph.

    Assortativity measures the similarity of connections
    in the graph with respect to the node degree.

    Arguments
    ----------
    adj : Scipy sparse adjacency matrix representing a graph

    Returns
    -------
    r : float
    Assortativity of graph by degree.

    Notes
    --------
    This is a faster implementation of 
    `networkx.degree_assortativity_coefficient`

    See Also
    --------
    graphgallery.degree_mixing_matrix
    networkx.degree_assortativity_coefficient


    References
    ----------
    .. [1] M. E. J. Newman, Mixing patterns in networks,
    Physical Review E, 67 026126, 2003
    .. [2] Foster, J.G., Foster, D.V., Grassberger, P. & Paczuski, M. 
    Edge direction and the structure of networks, PNAS 107, 10815-20 (2010).
    """
    M, mapping = degree_mixing_matrix(adj)
    return numeric_ac(M, mapping)
