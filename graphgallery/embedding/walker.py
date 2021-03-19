import random
import numpy as np
import networkx as nx
import scipy.sparse as sp

from numba import njit

__all__ = ["RandomWalker", "BiasedRandomWalker", "BiasedRandomWalkerAlias"]


@njit
def random_choice(arr, p):
    """Similar to `numpy.random.choice` and it suppors p=option in numba.
    refer to <https://github.com/numba/numba/issues/2539#issuecomment-507306369>

    Parameters
    ----------
    arr : 1-D array-like
    p : 1-D array-like
        The probabilities associated with each entry in arr

    Returns
    -------
    samples : ndarray
        The generated random samples
    """
    return arr[np.searchsorted(np.cumsum(p), np.random.random(), side="right")]


class RandomWalker:
    def __init__(self, walk_length: int = 80, walk_number: int = 10):
        self.walk_length = walk_length
        self.walk_number = walk_number

    def walk(self, graph: sp.csr_matrix):
        walks = self.random_walk(graph.indices,
                                 graph.indptr,
                                 walk_length=self.walk_length,
                                 walk_number=self.walk_number)
        return walks

    @staticmethod
    @njit
    def random_walk(indices,
                    indptr,
                    walk_length,
                    walk_number):
        N = len(indptr) - 1
        for _ in range(walk_number):
            for n in range(N):
                walk = [n]
                current_node = n
                for _ in range(walk_length - 1):
                    neighbors = indices[
                        indptr[current_node]:indptr[current_node + 1]]
                    if neighbors.size == 0:
                        break
                    current_node = np.random.choice(neighbors)
                    walk.append(current_node)

                yield walk


class BiasedRandomWalker:

    def __init__(self, walk_length: int = 80,
                 walk_number: int = 10,
                 p: float = 0.5,
                 q: float = 0.5):
        self.walk_length = walk_length
        self.walk_number = walk_number
        try:
            _ = 1 / p
        except ZeroDivisionError:
            raise ValueError("The value of p is too small or zero to be used in 1/p.")
        self.p = p
        try:
            _ = 1 / q
        except ZeroDivisionError:
            raise ValueError("The value of q is too small or zero to be used in 1/q.")
        self.q = q

    def walk(self, graph: sp.csr_matrix):
        walks = self.random_walk(graph.indices,
                                 graph.indptr,
                                 walk_length=self.walk_length,
                                 walk_number=self.walk_number,
                                 p=self.p,
                                 q=self.q)
        return walks

    @staticmethod
    @njit
    def random_walk(indices,
                    indptr,
                    walk_length,
                    walk_number,
                    p=0.5,
                    q=0.5):

        N = len(indptr) - 1
        for _ in range(walk_number):
            for n in range(N):
                walk = [n]
                current_node = n
                previous_node = N
                previous_node_neighbors = np.empty(0, dtype=np.int32)
                for _ in range(walk_length - 1):
                    neighbors = indices[indptr[current_node]:indptr[current_node + 1]]
                    if neighbors.size == 0:
                        break

                    probability = np.array([1 / q] * neighbors.size)
                    probability[previous_node == neighbors] = 1 / p

                    for i, nbr in enumerate(neighbors):
                        if np.any(nbr == previous_node_neighbors):
                            probability[i] = 1.

                    norm_probability = probability / np.sum(probability)
                    current_node = random_choice(neighbors, norm_probability)
                    walk.append(current_node)
                    previous_node_neighbors = neighbors
                    previous_node = current_node
                yield walk


class BiasedRandomWalkerAlias:

    def __init__(self, walk_length: int = 80,
                 walk_number: int = 10,
                 p: float = 0.5,
                 q: float = 0.5):
        self.walk_length = walk_length
        self.walk_number = walk_number
        try:
            _ = 1 / p
        except ZeroDivisionError:
            raise ValueError("The value of p is too small or zero to be used in 1/p.")
        self.p = p
        try:
            _ = 1 / q
        except ZeroDivisionError:
            raise ValueError("The value of q is too small or zero to be used in 1/q.")
        self.q = q

    def walk(self, graph: sp.csr_matrix):
        graph = nx.from_scipy_sparse_matrix(graph,
                                            create_using=nx.DiGraph)
        self.preprocess_transition_probs(graph)
        walks = self.random_walk(graph,
                                 self.alias_nodes,
                                 self.alias_edges,
                                 walk_length=self.walk_length,
                                 walk_number=self.walk_number)
        return walks

    @staticmethod
    def random_walk(graph,
                    alias_nodes,
                    alias_edges,
                    walk_length=80,
                    walk_number=10):

        for _ in range(walk_number):
            for n in graph.nodes():
                walk = [n]
                current_node = n
                for _ in range(walk_length - 1):
                    neighbors = list(graph.neighbors(current_node))
                    if len(neighbors) > 0:
                        if len(walk) == 1:
                            current_node = neighbors[alias_sample(
                                alias_nodes[current_node][0],
                                alias_nodes[current_node][1])]
                        else:
                            prev = walk[-2]
                            edge = (prev, current_node)
                            current_node = neighbors[alias_sample(
                                alias_edges[edge][0], alias_edges[edge][1])]
                    else:
                        break

                    walk.append(current_node)
                yield walk

    def get_alias_edge(self, graph, t, v):
        p = self.p
        q = self.q

        unnormalized_probs = []
        for x in graph.neighbors(v):
            weight = graph[v][x].get('weight', 1.0)  # w_vx

            if x == t:  # d_tx == 0
                unnormalized_probs.append(weight / p)
            elif graph.has_edge(x, t):  # d_tx == 1
                unnormalized_probs.append(weight)
            else:  # d_tx > 1
                unnormalized_probs.append(weight / q)

        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return create_alias_table(normalized_probs)

    def preprocess_transition_probs(self, graph):
        alias_nodes = {}
        for node in graph.nodes():
            unnormalized_probs = [graph[node][nbr].get('weight', 1.0)
                                  for nbr in graph.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = create_alias_table(normalized_probs)

        alias_edges = {}

        for edge in graph.edges():
            alias_edges[edge] = self.get_alias_edge(graph, edge[0], edge[1])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges


def create_alias_table(area_ratio):
    l = len(area_ratio)
    accept, alias = [0] * l, [0] * l
    small, large = [], []
    area_ratio_ = np.array(area_ratio) * l
    for i, prob in enumerate(area_ratio_):
        if prob < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        small_idx, large_idx = small.pop(), large.pop()
        accept[small_idx] = area_ratio_[small_idx]
        alias[small_idx] = large_idx
        area_ratio_[large_idx] = area_ratio_[large_idx] - (1 - area_ratio_[small_idx])
        if area_ratio_[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)

    while large:
        large_idx = large.pop()
        accept[large_idx] = 1
    while small:
        small_idx = small.pop()
        accept[small_idx] = 1

    return accept, alias


def alias_sample(accept, alias):
    N = len(accept)
    i = int(random.random() * N)
    r = random.random()
    if r < accept[i]:
        return i
    else:
        return alias[i]
