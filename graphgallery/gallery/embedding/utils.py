import gensim
import numpy as np
import scipy.sparse as sp
from gensim.models import Word2Vec as _Word2Vec
from distutils.version import LooseVersion


def normalized_laplacian_matrix(graph, r=-0.5):
    graph = graph + sp.eye(graph.shape[0], dtype=graph.dtype, format='csr')

    degree = graph.sum(1).A1
    degree_power = np.power(degree, r)
    graph = graph.tocoo(copy=False)
    graph.data *= degree_power[graph.row]
    graph.data *= degree_power[graph.col]
    graph = graph.tocsr(copy=False)
    return graph


class Word2Vec(_Word2Vec):
    """A compatible version of Word2Vec"""

    def __init__(self, sentences=None, sg=0, hs=0, alpha=0.025, iter=5, size=100, window=5, workers=3, negative=5, seed=None, **kwargs):
        if LooseVersion(gensim.__version__) <= LooseVersion("4.0.0"):
            super().__init__(sentences,
                             size=size,
                             window=window,
                             min_count=0,
                             alpha=alpha,
                             sg=sg,
                             workers=workers,
                             iter=iter,
                             negative=negative,
                             hs=hs,
                             compute_loss=True,
                             seed=seed, **kwargs)

        else:
            super().__init__(sentences,
                             vector_size=size,
                             window=window,
                             min_count=0,
                             alpha=alpha,
                             sg=sg,
                             workers=workers,
                             epochs=iter,
                             negative=negative,
                             hs=hs,
                             compute_loss=True,
                             seed=seed, **kwargs)

    def get_embedding(self):
        if LooseVersion(gensim.__version__) <= LooseVersion("4.0.0"):
            embedding = self.wv.vectors[np.fromiter(
                map(int, self.wv.index2word), np.int32).argsort()]
        else:
            embedding = self.wv.vectors[np.fromiter(
                map(int, self.wv.index_to_key), np.int32).argsort()]

        return embedding
