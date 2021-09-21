import math
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing


class GraRep:
    r"""An implementation of `"GraRep" <https://dl.acm.org/citation.cfm?id=2806512>`_
    from the CIKM '15 paper "GraRep: Learning Graph Representations with Global
    Structural Information". The procedure uses sparse truncated SVD to learn
    embeddings for the powers of the PMI matrix computed from powers of the
    normalized adjacency matrix.
    """

    def __init__(self, dimensions: int = 32, iteration: int = 10, order: int = 5, seed: int = None):
        self.dimensions = dimensions
        self.iterations = iteration
        self.order = order
        self.seed = seed

    def _create_base_matrix(self, graph):
        """
        Creating a tuple with the normalized adjacency matrix.
        """
        degree = graph.sum(1).A1
        D_inverse = sp.diags(1.0 / degree, format="csr")
        A_hat = D_inverse @ graph
        return (A_hat, A_hat)

    def _create_target_matrix(self):
        """
        Creating a log transformed target matrix.
        """
        self._A_tilde = (self._A_tilde @ self._A_hat).tocoo()
        scores = np.log(self._A_tilde.data) - math.log(self._A_tilde.shape[0])
        mask = scores < 0
        rows = self._A_tilde.row[mask]
        cols = self._A_tilde.col[mask]
        scores = scores[mask]
        target_matrix = sp.csr_matrix((scores, (rows, cols)),
                                      shape=self._A_tilde.shape,
                                      dtype=np.float32)

        return target_matrix

    def _create_single_embedding(self, target_matrix):
        """
        Fitting a single SVD embedding of a PMI matrix.
        """
        svd = TruncatedSVD(n_components=self.dimensions,
                           n_iter=self.iterations,
                           random_state=self.seed)
        svd.fit(target_matrix)
        embedding = svd.transform(target_matrix)
        self._embeddings.append(embedding)

    def fit(self, graph: sp.csr_matrix):
        """
        Fitting a GraRep model.
        """
        self._embeddings = []
        self._A_tilde, self._A_hat = self._create_base_matrix(graph)
        target_matrix = self._create_target_matrix()
        self._create_single_embedding(target_matrix)
        for step in range(self.order - 1):
            target_matrix = self._create_target_matrix()
            self._create_single_embedding(target_matrix)

    def get_embedding(self, normalize=True) -> np.array:
        """Getting the node embedding.
        """
        embedding = np.concatenate(self._embeddings, axis=1)
        if normalize:
            embedding = preprocessing.normalize(embedding)
        return embedding
