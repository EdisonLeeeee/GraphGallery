import random
import numpy as np
import networkx as nx
import scipy.sparse as sp

from numba import njit, prange, boolean


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
    sample : ndarray with 1 element
        The generated random sample
    """
    return arr[np.searchsorted(np.cumsum(p), np.random.random(), side="right")]


class RandomWalker:
    """ Fast first-order random walks in DeepWalk

    Parameters:
    -----------
    walk_number (int): Number of random walks. Default is 10.
    walk_length (int): Length of random walks. Default is 80.
    """

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
    @njit(nogil=True)
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
    """Biased second order random walks in Node2Vec.

    Parameters:
    -----------
    walk_number (int): Number of random walks. Default is 10.
    walk_length (int): Length of random walks. Default is 80.
    p (float): Return parameter (1/p transition probability) to move towards from previous node.
    q (float): In-out parameter (1/q transition probability) to move away from previous node.
    """

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
    @njit(nogil=False)
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
    """Motivated by `PecanPy: A parallelized, efficient, and accelerated node2vec(+) in Python`
    Github: `https://github.com/krishnanlab/PecanPy`.

    Parameters:
    -----------
    walk_number (int): Number of random walks. Default is 10.
    walk_length (int): Length of random walks. Default is 80.
    p (float): Return parameter (1/p transition probability) to move towards from previous node.
    q (float): In-out parameter (1/q transition probability) to move away from previous node.
    extend (bool): whether to use the extended version (`node2vec`). See below.    
    mode (str): different modes including `PreComp` and `SparseOTF`. See below.


    Specify `extend=True` for using node2vec+, which is a natural extension of
    node2vec and handles weighted graph more effectively. For more information, see
    `Accurately Modeling Biased Random Walks on Weighted Wraphs Using Node2vec+`(https://arxiv.org/abs/2109.08031)


    `BiasedRandomWalkerAlias` operates in three different modes – PreComp and SparseOTF – that are optimized for networks of different sizes and densities:
    - `PreComp` for networks that are small (≤10k nodes; any density),
    - `SparseOTF` for networks that are large and sparse (>10k nodes; ≤10% of edges),
    These modes appropriately take advantage of compact/dense graph data structures, precomputing transition probabilities, and computing 2nd-order transition probabilities during walk generation to achieve significant improvements in performance.

    """

    def __init__(self, walk_length: int = 80,
                 walk_number: int = 10,
                 p: float = 0.5,
                 q: float = 0.5,
                 extend: bool = True,
                 mode='PreComp'):
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

        self.extend = extend

        if mode == 'PreComp':
            self.get_move_forward = get_move_forward_PreComp
        elif mode == 'SparseOTF':
            self.get_move_forward = get_move_forward_SparseOTF
        else:
            raise ValueError("`mode` should be one of 'PreComp' and 'SparseOTF'")

    def walk(self, graph: sp.csr_matrix):
        """Generate walks starting from each nodes ``walk_number`` time.

            Note:
            ----------
            This is the master process that spawns worker processes, where the
            worker function ``random_walk`` genearte a single random walk
            starting from a vertex of the graph.

            Parameters
            ----------
            graph: scipy.sparse.csr_matrix, the input graph

        """
        self.preprocess_transition_probs(graph)
        walk_number = self.walk_number
        walk_length = self.walk_length
        num_nodes = graph.shape[0]
        nodes = np.arange(num_nodes, dtype=np.int32)
        start_nodes = np.concatenate([nodes] * walk_number)
        np.random.shuffle(start_nodes)

        move_forward = self.get_move_forward(self, graph)
        has_nbrs = get_has_nbrs(graph)

        @njit(parallel=True, nogil=True)
        def random_walk():
            """Simulate a random walk starting from start node."""
            n = start_nodes.size
            # use last entry of each walk index array to keep track of effective walk length
            walk_idx_mat = np.zeros((n, walk_length + 2), dtype=np.int32)
            walk_idx_mat[:, 0] = start_nodes  # initialize seeds
            walk_idx_mat[:, -1] = walk_length + 1  # set to full walk length by default

            for i in prange(n):
                # initialize first step as normal random walk
                start_node_idx = walk_idx_mat[i, 0]
                if has_nbrs(start_node_idx):
                    walk_idx_mat[i, 1] = move_forward(start_node_idx)
                else:
                    walk_idx_mat[i, -1] = 1
                    continue

                # start bias random walk
                for j in range(2, walk_length + 1):
                    cur_idx = walk_idx_mat[i, j - 1]
                    if has_nbrs(cur_idx):
                        prev_idx = walk_idx_mat[i, j - 2]
                        walk_idx_mat[i, j] = move_forward(cur_idx, prev_idx)
                    else:
                        walk_idx_mat[i, -1] = j
                        break

            return walk_idx_mat

        walks = [list(map(str, walk[:walk[-1]])) for walk in random_walk()]

        return walks

    def preprocess_transition_probs(self, graph):
        """Precompute and store 2nd order transition probabilities."""
        data = graph.data
        indices = graph.indices
        indptr = graph.indptr
        p = self.p
        q = self.q

        num_nodes = graph.shape[0]  # number of nodes
        num_nbrs = indptr[1:] - indptr[:-1]  # number of nbrs per node
        num_nbrs_2nd = np.power(num_nbrs, 2)  # number of 2nd order trans probs per node

        if self.extend:
            compute_fn = get_normalized_probs_extended
            deg = graph.sum(1).A1
            avg_wts = deg / num_nbrs  # average edge weights
        else:
            compute_fn = get_normalized_probs
            avg_wts = None

        alias_dim = num_nbrs
        # use 64 bit unsigned int to prevent overfloating of alias_indptr
        alias_indptr = np.zeros(indptr.size, dtype=np.uint64)
        alias_indptr[1:] = np.cumsum(num_nbrs_2nd)
        n_probs = alias_indptr[-1]  # total number of 2nd order transition probs

        @njit(parallel=True, nogil=True)
        def compute_all_transition_probs():
            alias_j = np.zeros(n_probs, dtype=np.int32)
            alias_q = np.zeros(n_probs, dtype=np.float64)

            for idx in range(num_nodes):
                offset = alias_indptr[idx]
                dim = alias_dim[idx]

                nbrs = indices[indptr[idx]: indptr[idx + 1]]
                for nbr_idx in prange(num_nbrs[idx]):
                    nbr = nbrs[nbr_idx]
                    probs = compute_fn(data, indices, indptr, p, q, idx, nbr, avg_wts)

                    start = offset + dim * nbr_idx
                    j_tmp, q_tmp = alias_setup(probs)

                    for i in range(dim):
                        alias_j[start + i] = j_tmp[i]
                        alias_q[start + i] = q_tmp[i]

            return alias_j, alias_q

        alias_j, alias_q = compute_all_transition_probs()

        self.alias_j = alias_j
        self.alias_q = alias_q
        self.alias_dim = alias_dim
        self.alias_indptr = alias_indptr


def get_move_forward_PreComp(self, graph):
    """Wrap ``move_forward``.

    This function returns a ``numba.jit`` compiled function that takes
    current vertex index (and the previous vertex index if available) and
    return the next vertex index by sampling from a discrete random
    distribution based on the transition probabilities that are read off
    the precomputed transition probabilities table.

    Note:
        The returned function is used by the ``walk`` method.

    """
    alias_j = self.alias_j
    alias_q = self.alias_q
    alias_dim = self.alias_dim
    alias_indptr = self.alias_indptr
    p, q = self.p, self.q
    data = graph.data
    indices = graph.indices
    indptr = graph.indptr

    compute_fn = get_normalized_probs

    @njit(nogil=True)
    def move_forward(cur_idx, prev_idx=None):
        """Move to next node based on transition probabilities."""
        if prev_idx is None:
            normalized_probs = compute_fn(data, indices, indptr, p, q, cur_idx, None, None)
            cdf = np.cumsum(normalized_probs)
            choice = np.searchsorted(cdf, np.random.random())
        else:
            # find index of neighbor for reading alias
            start = indptr[cur_idx]
            end = indptr[cur_idx + 1]
            nbr_idx = np.searchsorted(indices[start:end], prev_idx)
            if indices[start + nbr_idx] != prev_idx:
                raise RuntimeError("FATAL ERROR! Neighbor not found.")

            dim = alias_dim[cur_idx]
            start = alias_indptr[cur_idx] + dim * nbr_idx
            end = start + dim
            choice = alias_draw(alias_j[start:end], alias_q[start:end])

        return indices[indptr[cur_idx] + choice]

    return move_forward


def get_move_forward_SparseOTF(self, graph):
    """Wrap ``move_forward``.

    This function returns a ``numba.jit`` compiled function that takes
    current vertex index (and the previous vertex index if available) and
    return the next vertex index by sampling from a discrete random
    distribution based on the transition probabilities that are calculated
    on-the-fly.

    Note:
        The returned function is used by the ``walk`` method.

    """
    p, q = self.p, self.q
    data = graph.data
    indices = graph.indices
    indptr = graph.indptr

    if self.extend:
        compute_fn = get_normalized_probs_extended
        deg = graph.sum(1).A1
        num_nbrs = indptr[1:] - indptr[:-1]  # number of nbrs per node
        avg_wts = deg / num_nbrs  # average edge weights
    else:
        compute_fn = get_normalized_probs
        avg_wts = None

    @njit(nogil=True)
    def move_forward(cur_idx, prev_idx=None):
        """Move to next node."""
        normalized_probs = compute_fn(
            # data, indices, indptr, p, q, cur_idx, prev_idx)
            data, indices, indptr, p, q, cur_idx, prev_idx, avg_wts)
        cdf = np.cumsum(normalized_probs)
        choice = np.searchsorted(cdf, np.random.random())

        return indices[indptr[cur_idx] + choice]

    return move_forward


@njit(nogil=True)
def get_normalized_probs(data, indices, indptr, p, q, cur_idx, prev_idx, avg_wts):
    """Calculate node2vec transition probabilities.

    Calculate 2nd order transition probabilities by first finidng the
    neighbors of the current state that are not reachable from the previous
    state, and devide the according edge weights by the in-out parameter
    ``q``. Then devide the edge weight from previous state by the return
    parameter ``p``. Finally, the transition probabilities are computed by
    normalizing the biased edge weights.

    Note:
        If ``prev_idx`` present, calculate 2nd order biased transition,
    otherwise calculate 1st order transition.

    """
    def get_nbrs_idx(idx):
        return indices[indptr[idx]: indptr[idx + 1]]

    def get_nbrs_weight(idx):
        return data[indptr[idx]: indptr[idx + 1]].copy()

    nbrs_idx = get_nbrs_idx(cur_idx)
    unnormalized_probs = get_nbrs_weight(cur_idx)

    if prev_idx is not None:  # 2nd order biased walk
        prev_ptr = np.where(nbrs_idx == prev_idx)[0]  # find previous state index
        src_nbrs_idx = get_nbrs_idx(prev_idx)  # neighbors of previous state
        non_com_nbr = isnotin(nbrs_idx, src_nbrs_idx)  # neighbors of current but not previous
        non_com_nbr[prev_ptr] = False  # exclude previous state from out biases

        unnormalized_probs[non_com_nbr] /= q  # apply out biases
        unnormalized_probs[prev_ptr] /= p  # apply the return bias

    normalized_probs = unnormalized_probs / unnormalized_probs.sum()

    return normalized_probs


@njit(nogil=True)
def get_normalized_probs_extended(data, indices, indptr, p, q, cur_idx, prev_idx, average_weight_ary):
    """Calculate node2vec+ transition probabilities."""
    def get_nbrs_idx(idx):
        return indices[indptr[idx]: indptr[idx + 1]]

    def get_nbrs_weight(idx):
        return data[indptr[idx]: indptr[idx + 1]].copy()

    nbrs_idx = get_nbrs_idx(cur_idx)
    unnormalized_probs = get_nbrs_weight(cur_idx)

    if prev_idx is not None:  # 2nd order biased walk
        prev_ptr = np.where(nbrs_idx == prev_idx)[0]  # find previous state index
        src_nbrs_idx = get_nbrs_idx(prev_idx)  # neighbors of previous state
        out_ind, t = isnotin_extended(nbrs_idx, src_nbrs_idx,
                                      get_nbrs_weight(prev_idx),
                                      average_weight_ary)  # determine out edges
        out_ind[prev_ptr] = False  # exclude previous state from out biases

        # compute out biases
        alpha = (1 / q + (1 - 1 / q) * t[out_ind])

        # surpress noisy edges
        alpha[unnormalized_probs[out_ind] < average_weight_ary[cur_idx]] = np.minimum(1, 1 / q)
        unnormalized_probs[out_ind] *= alpha  # apply out biases
        unnormalized_probs[prev_ptr] /= p  # apply the return bias

    normalized_probs = unnormalized_probs / unnormalized_probs.sum()

    return normalized_probs


@njit(nogil=True)
def alias_setup(probs):
    """Construct alias lookup table.

    This code is modified from the blog post here:
    https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    , where you can find more details about how the method work. In general,
    the alias method improves the time complexity of sampling from a discrete
    random distribution to O(1) if the alias table is setup in advance.

    Parameters:
    -----------
    probs (list(float64)): normalized transition probabilities array, could
        be in either list or numpy.ndarray, of float64 values.

    """
    k = probs.size
    q = np.zeros(k, dtype=np.float64)
    j = np.zeros(k, dtype=np.int32)

    smaller = np.zeros(k, dtype=np.int32)
    larger = np.zeros(k, dtype=np.int32)
    smaller_ptr = 0
    larger_ptr = 0

    for kk in range(k):
        q[kk] = k * probs[kk]
        if q[kk] < 1.0:
            smaller[smaller_ptr] = kk
            smaller_ptr += 1
        else:
            larger[larger_ptr] = kk
            larger_ptr += 1

    while (smaller_ptr > 0) & (larger_ptr > 0):
        smaller_ptr -= 1
        small = smaller[smaller_ptr]
        larger_ptr -= 1
        large = larger[larger_ptr]

        j[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller[smaller_ptr] = large
            smaller_ptr += 1
        else:
            larger[larger_ptr] = large
            larger_ptr += 1

    return j, q


@njit(nogil=True)
def alias_draw(j, q):
    """Draw sample from a non-uniform discrete distribution using alias sampling."""
    k = j.size

    kk = np.random.randint(k)
    if np.random.rand() < q[kk]:
        return kk
    else:
        return j[kk]


def get_has_nbrs(graph):
    """Wrap ``has_nbrs``."""
    indptr = graph.indptr

    @njit(nogil=True)
    def has_nbrs(idx):
        return indptr[idx] != indptr[idx + 1]

    return has_nbrs


@njit(nogil=True)
def isnotin(ptr_ary1, ptr_ary2):
    """Find node2vec out edges.

    The node2vec out edges is determined by non-common neighbors. This function
    find out neighbors of node1 that are not neighbors of node2, by picking out
    values in ``ptr_ary1`` but not in ``ptr_ary2``, which correspond to the
    neighbor pointers for the current state and the previous state, resp.

    Note:
        This function does not remove the index of the previous state. Instead,
    the index of the previous state will be removed once the indicator is
    returned to the ``get_normalized_probs``.

    Parameters:
    -----------
    ptr_ary1 (:obj:`numpy.ndarray` of :obj:`int32`): array of pointers to
        the neighbors of the current state
    ptr_ary2 (:obj:`numpy.ndarray` of :obj:`int32`): array of pointers to
        the neighbors of the previous state

    Returns:
    -----------
    Indicator of whether a neighbor of the current state is considered as
        an "out edge"

    Example:
    -----------
    The values in the two neighbor pointer arrays are sorted ascendingly.
    The main idea is to scan through ``ptr_ary1`` and compare the values in
    ``ptr_ary2``. In this way, at most one pass per array is needed to find
    out the non-common neighbor pointers instead of a nested loop (for each
    element in ``ptr_ary1``, compare against every element in``ptr_ary2``),
    which is much slower. Checkout the following example for more intuition.
    The ``*`` above ``ptr_ary1`` and ``ptr_ary2`` indicate the indices
    ``idx1`` and ``idx2``, respectively, which keep track of the scaning
    progress.

    >>> ptr_ary1 = [1, 2, 5]
    >>> ptr_ary2 = [1, 5]
    >>>
    >>> # iteration1: indicator = [False, True, True]
    >>>  *
    >>> [1, 2, 5]
    >>>  *
    >>> [1, 5]
    >>>
    >>> # iteration2: indicator = [False, True, True]
    >>>     *
    >>> [1, 2, 5]
    >>>     *
    >>> [1, 5]
    >>>
    >>> # iteration3: indicator = [False, True, False]
    >>>        *
    >>> [1, 2, 5]
    >>>     *
    >>> [1, 5]
    >>>
    >>> # end of loop

    """
    indicator = np.ones(ptr_ary1.size, dtype=boolean)
    idx2 = 0
    for idx1 in range(ptr_ary1.size):
        if idx2 == ptr_ary2.size:  # end of ary2
            break

        ptr1 = ptr_ary1[idx1]
        ptr2 = ptr_ary2[idx2]

        if ptr1 < ptr2:
            continue

        elif ptr1 == ptr2:  # found a matching value
            indicator[idx1] = False
            idx2 += 1

        elif ptr1 > ptr2:
            # sweep through ptr_ary2 until ptr2 catch up on ptr1
            for j in range(idx2, ptr_ary2.size):
                ptr2 = ptr_ary2[j]
                if ptr2 == ptr1:
                    indicator[idx1] = False
                    idx2 = j + 1
                    break

                elif ptr2 > ptr1:
                    idx2 = j
                    break

    return indicator


@njit(nogil=True)
def isnotin_extended(ptr_ary1, ptr_ary2, wts_ary2, avg_wts):
    """Find node2vec+ out edges.

    The node2vec+ out edges is determined by considering the edge weights
    connecting node2 (the potential next state) to the previous state. Unlinke
    node2vec, which only considers neighbors of current state that are not
    neighbors of the previous state, node2vec+ also considers neighbors of
    the previous state as out edges if the edge weight is below average.

    Parameters:
    -----------
    ptr_ary1 (:obj:`numpy.ndarray` of :obj:`uint32`): array of pointers to
        the neighbors of the current state
    ptr_ary2 (:obj:`numpy.ndarray` of :obj:`uint32`): array of pointers to
        the neighbors of the previous state
    wts_ary2 (:obj: `numpy.ndarray` of :obj:`float64`): array of edge
        weights of the previous state
    avg_wts (:obj: `numpy.ndarray` of :obj:`float64`): array of average
        edge weights of each node

    Return:
    -----------
    Indicator of whether a neighbor of the current state is considered as
        an "out edge", with the corresponding parameters used to fine tune
        the out biases

    """
    indicator = np.ones(ptr_ary1.size, dtype=boolean)
    t = np.zeros(ptr_ary1.size, dtype=np.float64)
    idx2 = 0
    for idx1 in range(ptr_ary1.size):
        if idx2 == ptr_ary2.size:  # end of ary2
            break

        ptr1 = ptr_ary1[idx1]
        ptr2 = ptr_ary2[idx2]

        if ptr1 < ptr2:
            continue

        elif ptr1 == ptr2:  # found a matching value
            if wts_ary2[idx2] >= avg_wts[ptr2]:  # check if loose
                indicator[idx1] = False
            else:
                t[idx1] = wts_ary2[idx2] / avg_wts[ptr2]
            idx2 += 1

        elif ptr1 > ptr2:
            # sweep through ptr_ary2 until ptr2 catch up on ptr1
            for j in range(idx2, ptr_ary2.size):
                ptr2 = ptr_ary2[j]
                if ptr2 == ptr1:
                    if wts_ary2[j] >= avg_wts[ptr2]:
                        indicator[idx1] = False
                    else:
                        t[idx1] = wts_ary2[j] / avg_wts[ptr2]
                    idx2 = j + 1
                    break

                elif ptr2 > ptr1:
                    idx2 = j
                    break

    return indicator, t

##############################################################
# The following method is deprecated since we got a better implementation using numba
##############################################################
# class BiasedRandomWalkerAlias:

#     def __init__(self, walk_length: int = 80,
#                  walk_number: int = 10,
#                  p: float = 0.5,
#                  q: float = 0.5):
#         self.walk_length = walk_length
#         self.walk_number = walk_number
#         try:
#             _ = 1 / p
#         except ZeroDivisionError:
#             raise ValueError("The value of p is too small or zero to be used in 1/p.")
#         self.p = p
#         try:
#             _ = 1 / q
#         except ZeroDivisionError:
#             raise ValueError("The value of q is too small or zero to be used in 1/q.")
#         self.q = q

#     def walk(self, graph: sp.csr_matrix):
#         graph = nx.from_scipy_sparse_matrix(graph,
#                                             create_using=nx.DiGraph)
#         self.preprocess_transition_probs(graph)
#         walks = self.random_walk(graph,
#                                  self.alias_nodes,
#                                  self.alias_edges,
#                                  walk_length=self.walk_length,
#                                  walk_number=self.walk_number)
#         return walks

#     @staticmethod
#     def random_walk(graph,
#                     alias_nodes,
#                     alias_edges,
#                     walk_length=80,
#                     walk_number=10):

#         for _ in range(walk_number):
#             for n in graph.nodes():
#                 walk = [n]
#                 current_node = n
#                 for _ in range(walk_length - 1):
#                     neighbors = list(graph.neighbors(current_node))
#                     if len(neighbors) > 0:
#                         if len(walk) == 1:
#                             current_node = neighbors[alias_sample(
#                                 alias_nodes[current_node][0],
#                                 alias_nodes[current_node][1])]
#                         else:
#                             prev = walk[-2]
#                             edge = (prev, current_node)
#                             current_node = neighbors[alias_sample(
#                                 alias_edges[edge][0], alias_edges[edge][1])]
#                     else:
#                         break

#                     walk.append(current_node)
#                 yield walk

#     def get_alias_edge(self, graph, t, v):
#         p = self.p
#         q = self.q

#         unnormalized_probs = []
#         for x in graph.neighbors(v):
#             weight = graph[v][x].get('weight', 1.0)  # w_vx

#             if x == t:  # d_tx == 0
#                 unnormalized_probs.append(weight / p)
#             elif graph.has_edge(x, t):  # d_tx == 1
#                 unnormalized_probs.append(weight)
#             else:  # d_tx > 1
#                 unnormalized_probs.append(weight / q)

#         norm_const = sum(unnormalized_probs)
#         normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

#         return create_alias_table(normalized_probs)

#     def preprocess_transition_probs(self, graph):
#         alias_nodes = {}
#         for node in graph.nodes():
#             unnormalized_probs = [graph[node][nbr].get('weight', 1.0)
#                                   for nbr in graph.neighbors(node)]
#             norm_const = sum(unnormalized_probs)
#             normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
#             alias_nodes[node] = create_alias_table(normalized_probs)

#         alias_edges = {}

#         for edge in graph.edges():
#             alias_edges[edge] = self.get_alias_edge(graph, edge[0], edge[1])

#         self.alias_nodes = alias_nodes
#         self.alias_edges = alias_edges


# def create_alias_table(area_ratio):
#     l = len(area_ratio)
#     accept, alias = [0] * l, [0] * l
#     small, large = [], []
#     area_ratio_ = np.array(area_ratio) * l
#     for i, prob in enumerate(area_ratio_):
#         if prob < 1.0:
#             small.append(i)
#         else:
#             large.append(i)

#     while small and large:
#         small_idx, large_idx = small.pop(), large.pop()
#         accept[small_idx] = area_ratio_[small_idx]
#         alias[small_idx] = large_idx
#         area_ratio_[large_idx] = area_ratio_[large_idx] - (1 - area_ratio_[small_idx])
#         if area_ratio_[large_idx] < 1.0:
#             small.append(large_idx)
#         else:
#             large.append(large_idx)

#     while large:
#         large_idx = large.pop()
#         accept[large_idx] = 1
#     while small:
#         small_idx = small.pop()
#         accept[small_idx] = 1

#     return accept, alias


# def alias_sample(accept, alias):
#     N = len(accept)
#     i = int(random.random() * N)
#     r = random.random()
#     if r < accept[i]:
#         return i
#     else:
#         return alias[i]
