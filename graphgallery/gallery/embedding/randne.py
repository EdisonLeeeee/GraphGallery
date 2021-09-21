import numpy as np
import scipy.sparse as sp
from sklearn import preprocessing


class RandNE:
    r"""An implementation of `"RandNE" <https://zw-zhang.github.io/files/2018_ICDM_RandNE.pdf>`_ from the ICDM '18 paper "Billion-scale Network Embedding with Iterative Random Projection". The procedure uses normalized adjacency matrix based
    smoothing on an orthogonalized random normally generate base node embedding matrix.
    """

    def __init__(self, dimensions: int = 128, alphas: list = [0.5, 0.5], seed: int = None):
        self.dimensions = dimensions
        self.alphas = alphas
        self.seed = seed

    def _create_smoothing_matrix(self, graph):
        """
        Creating the normalized adjacency matrix.
        """
        degree = graph.sum(1).A1
        D_inverse = sp.diags(1.0 / degree)
        A_hat = D_inverse @ graph
        return A_hat

    def _create_embedding(self, A_hat):
        """
        Using the random orthogonal smoothing.
        """
        sd = 1 / self.dimensions
        base_embedding = np.random.normal(0, sd, (A_hat.shape[0], self.dimensions))
        base_embedding, _ = np.linalg.qr(base_embedding)
        embedding = np.zeros(base_embedding.shape)
        alpha_sum = sum(self.alphas)
        for alpha in self.alphas:
            base_embedding = A_hat.dot(base_embedding)
            embedding = embedding + alpha * base_embedding
        embedding = embedding / alpha_sum
        return embedding

    def fit(self, graph: sp.csr_matrix):
        """
        Fitting a NetMF model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        A_hat = self._create_smoothing_matrix(graph)
        self._embedding = self._create_embedding(A_hat)

    def get_embedding(self, normalize=False) -> np.array:
        """Getting the node embedding."""
        embedding = self._embedding
        if normalize:
            embedding = preprocessing.scale(embedding)
        return embedding
