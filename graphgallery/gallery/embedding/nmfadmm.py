import numpy as np
import scipy.sparse as sp
from sklearn import preprocessing


class NMFADMM:
    r"""An implementation of `"NMF-ADMM" <http://statweb.stanford.edu/~dlsun/papers/nmf_admm.pdf>`_
    from the ICASSP '14 paper "Alternating Direction Method of Multipliers for 
    Non-Negative Matrix Factorization with the Beta-Divergence". The procedure
    learns an embedding of the normalized adjacency matrix with by using the alternating
    direction method of multipliers to solve a non negative matrix factorization problem.
    """

    def __init__(self, dimensions: int = 32, iterations: int = 100, rho: float = 1.0, seed: int = None):
        self.dimensions = dimensions
        self.iterations = iterations
        self.rho = rho
        self.seed = seed

    def _init_weights(self):
        """
        Initializing model weights.
        """
        self._W = np.random.uniform(-0.1, 0.1, (self._V.shape[0], self.dimensions))
        self._H = np.random.uniform(-0.1, 0.1, (self.dimensions, self._V.shape[1]))
        X_i, Y_i = np.nonzero(self._V)
        scores = self._W[X_i] * self._H[:, Y_i].T + np.random.uniform(0, 1, (self.dimensions, ))
        values = np.sum(scores, axis=-1)
        self._X = sp.coo_matrix((values, (X_i, Y_i)), shape=self._V.shape)
        self._W_plus = np.random.uniform(0, 0.1, (self._V.shape[0], self.dimensions))
        self._H_plus = np.random.uniform(0, 0.1, (self.dimensions, self._V.shape[1]))
        self._alpha_X = sp.coo_matrix((np.zeros(values.shape[0]), (X_i, Y_i)), shape=self._V.shape)
        self._alpha_W = np.zeros(self._W.shape)
        self._alpha_H = np.zeros(self._H.shape)

    def _update_W(self):
        """
        Updating user_1 matrix.
        """
        left = np.linalg.pinv(self._H.dot(self._H.T) + np.eye(self.dimensions))
        right_1 = self._X.dot(self._H.T).T + self._W_plus.T
        right_2 = (1.0 / self.rho) * (self._alpha_X.dot(self._H.T).T - self._alpha_W.T)
        self.W = left.dot(right_1 + right_2).T

    def _update_H(self):
        """
        Updating user_2 matrix.
        """
        left = np.linalg.pinv(self._W.T.dot(self._W) + np.eye(self.dimensions))
        right_1 = self._X.T.dot(self._W).T + self._H_plus
        right_2 = (1.0 / self.rho) * (self._alpha_X.T.dot(self._W).T - self._alpha_H)
        self._H = left.dot(right_1 + right_2)

    def _update_X(self):
        """
        Updating user_1-user_2 matrix.
        """
        iX, iY = np.nonzero(self._V)
        values = np.sum(self._W[iX] * self._H[:, iY].T, axis=-1)
        scores = sp.coo_matrix((values - 1, (iX, iY)), shape=self._V.shape)
        left = self.rho * scores - self._alpha_X
        right = (left.power(2) + 4.0 * self.rho * self._V).power(0.5)
        self._X = (left + right) / (2 * self.rho)

    def _update_W_plus(self):
        """
        Updating positive primal user_1 factors.
        """
        self._W_plus = np.maximum(self._W + (1 / self.rho) * self._alpha_W, 0)

    def _update_H_plus(self):
        """
        Updating positive primal user_2 factors.
        """
        self._H_plus = np.maximum(self._H + (1 / self.rho) * self._alpha_H, 0)

    def _update_alpha_X(self):
        """
        Updating target matrix dual.
        """
        iX, iY = np.nonzero(self._V)
        values = np.sum(self._W[iX] * self._H[:, iY].T, axis=-1)
        scores = sp.coo_matrix((values, (iX, iY)), shape=self._V.shape)
        self._alpha_X = self._alpha_X + self.rho * (self._X - scores)

    def _update_alpha_W(self):
        """
        Updating user dual factors.
        """
        self._alpha_W = self._alpha_W + self.rho * (self._W - self._W_plus)

    def _update_alpha_H(self):
        """
        Updating item dual factors.
        """
        self._alpha_H = self._alpha_H + self.rho * (self._H - self._H_plus)

    def _create_base_matrix(self, graph):
        """
        Creating the normalized adjacency matrix.
        """
        degree = graph.sum(1).A1
        D_inverse = sp.diags(1.0 / degree, format="csr")
        A_hat = D_inverse @ graph
        return A_hat

    def fit(self, graph: sp.csr_matrix):
        """
        Fitting an NMF model on the normalized adjacency matrix with ADMM.
        """
        self._V = self._create_base_matrix(graph)
        self._init_weights()
        for _ in range(self.iterations):
            self._update_W()
            self._update_H()
            self._update_X()
            self._update_W_plus()
            self._update_H_plus()
            self._update_alpha_X()
            self._update_alpha_W()
            self._update_alpha_H()

    def get_embedding(self, normalize=True) -> np.array:
        """Getting the node embedding."""
        embedding = np.concatenate([self._W_plus, self._H_plus.T], axis=1)
        if normalize:
            embedding = preprocessing.normalize(embedding)
        return embedding
