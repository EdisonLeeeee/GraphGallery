import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing


class NetMF:
    r"""An implementation of `"NetMF" <https://keg.cs.tsinghua.edu.cn/jietang/publications/WSDM18-Qiu-et-al-NetMF-network-embedding.pdf>`_
    from the WSDM '18 paper "Network Embedding as Matrix Factorization: Unifying
    DeepWalk, LINE, PTE, and Node2Vec". The procedure uses sparse truncated SVD to
    learn embeddings for the pooled powers of the PMI matrix computed from powers
    of the normalized adjacency matrix.
    """

    def __init__(self, dimensions: int = 16, iteration: int = 10, order: int = 2,
                 negative_samples: int = 3, seed: int = None):
        self.dimensions = dimensions
        self.iterations = iteration
        self.order = order
        self.negative_samples = negative_samples
        self.seed = seed

    def _create_base_matrix(self, graph):
        """
        Creating the normalized adjacency matrix.
        """
        degree = graph.sum(1).A1
        D_inverse = sp.diags(1.0 / degree, format="csr")
        A_hat = D_inverse @ graph
        return (A_hat, A_hat, A_hat, D_inverse)

    def _create_target_matrix(self, graph):
        """
        Creating a log transformed target matrix.
        """
        A_pool, A_tilde, A_hat, D_inverse = self._create_base_matrix(graph)
        for _ in range(self.order - 1):
            A_tilde = sp.coo_matrix(A_tilde.dot(A_hat))
            A_pool = A_pool + A_tilde
        A_pool = (graph.nnz * A_pool) / (self.order * self.negative_samples)
        A_pool = sp.coo_matrix(A_pool.dot(D_inverse))
        A_pool.data[A_pool.data < 1.0] = 1.0
        target_matrix = sp.coo_matrix((np.log(A_pool.data), (A_pool.row, A_pool.col)),
                                      shape=A_pool.shape,
                                      dtype=np.float32)
        return target_matrix

    def _create_embedding(self, target_matrix):
        """
        Fitting a truncated SVD embedding of a PMI matrix.
        """
        svd = TruncatedSVD(n_components=self.dimensions,
                           n_iter=self.iterations,
                           random_state=self.seed)
        svd.fit(target_matrix)
        embedding = svd.transform(target_matrix)
        return embedding

    def fit(self, graph: sp.csr_matrix):
        """
        Fitting a NetMF model.
        """
        target_matrix = self._create_target_matrix(graph)
        self._embedding = self._create_embedding(target_matrix)

    def get_embedding(self, normalize=True) -> np.array:
        """Getting the node embedding."""
        embedding = self._embedding
        if normalize:
            embedding = preprocessing.normalize(embedding)
        return embedding
