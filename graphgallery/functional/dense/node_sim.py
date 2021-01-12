import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from itertools import product


def knn_graph(x, k=20):
    X = np.zeros_like(x)
    X[x != 0] = 1
    sims = cosine_similarity(X)
    sims[np.diag_indices(len(sims))] = 0.

    for i, sim in enumerate(sims):
        indices_argsort = np.argsort(sim)
        sims[i, indices_argsort[: -k]] = 0.

    adj_knn = sp.csr_matrix(sims)
    return adj_knn


def attr_sim(x, k=5):
    X = np.zeros_like(x)
    X[x != 0] = 1.

    sims = cosine_similarity(X)
    indices_sorted = sims.argsort(1)
    selected = np.hstack((indices_sorted[:, :k],
                          indices_sorted[:, - k - 1:]))

    selected_set = set()
    for i in range(len(sims)):
        for pair in product([i], selected[i]):
            if pair[0] > pair[1]:
                pair = (pair[1], pair[0])
            if pair[0] == pair[1]:
                continue
            selected_set.add(pair)

    sampled = np.transpose(list(selected_set))
#     print('number of sampled:', len(sampled[0]))
    node_pairs = (sampled[0], sampled[1])
    return sims[node_pairs], node_pairs
