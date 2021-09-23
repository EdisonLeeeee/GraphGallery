import numpy as np
import numba as nb
import scipy.sparse as sp
from numpy.linalg import inv
from tqdm import tqdm
from .trainer import Trainer


class BANE(Trainer):
    r"""An implementation of `"BANE" <https://shiruipan.github.io/publication/yang-binarized-2018/yang-binarized-2018.pdf>`_
    from the ICDM '18 paper "Binarized Attributed Network Embedding". The
    procedure first calculates the truncated SVD of an adjacency - feature matrix
    product. This matrix is further decomposed by a binary CCD based technique.

    The reference MatLab implementation is available <https://github.com/ICDM2018-BANE/BANE>.
    """

    def __init__(self, dimensions: int = 32,
                 gamma: float = 0.7,
                 order: int = 4,
                 alpha: float = 0.001,
                 epochs: int = 20,
                 binary_iterations: int = 20,
                 name: str = None,
                 seed: int = None):

        kwargs = locals()
        kwargs.pop("self")
        super().__init__(**kwargs)

    def fit(self, graph: sp.csr_matrix, x: np.ndarray):

        P = self.get_P(graph, x)
        P = self.svd(P, self.dimensions)
        self._embedding = self.binary_optimize(P)

    def get_P(self, A, x):
        I = sp.eye(A.shape[0], format='csr')
        A = A + I
        deg = A.sum(1).A1
        D = sp.diags(deg)
        DN = sp.diags(1 / deg)
        L = D - A
        P = I - self.gamma * DN @ L
        P_power = P
        for _ in range(self.order - 1):
            P_power = P_power @ P

        Z = sp.diags(np.power((x * x).sum(1), -0.5))
        x = Z.T.dot(x)
        return P_power @ x

    @staticmethod
    def svd(P, dimensions):
        """
        Reducing the dimensionality with SVD in the 1st step.
        """
        U, S, V = sp.linalg.svds(P, k=dimensions)
        return U * S

    def binary_optimize(self, P):
        """
        Starting 2nd optimization phase with power iterations and CCD.
        """
        binary_iterations = self.binary_iterations
        alpha = self.alpha
        B = np.sign(np.random.normal(size=(P.shape[0], self.dimensions)))
        for _ in tqdm(range(self.epochs)):
            G = self.update_G(B, P, alpha=alpha)
            Q = self.update_Q(G, P)
            B = self.update_B(B, Q, G, binary_iterations=binary_iterations)
        return B

    @staticmethod
    def update_G(B, P, alpha=0.001):
        """
        Updating the kernel matrix.
        """
        G = np.dot(B.transpose(), B)
        G = G + alpha * np.eye(B.shape[-1])
        G = inv(G)
        G = G.dot(B.transpose()).dot(P)
        return G

    @staticmethod
    def update_Q(G, P):
        """
        Updating the rescaled target matrix.
        """
        Q = G.dot(P.transpose()).transpose()
        return Q

    @staticmethod
    @nb.njit(parallel=True, nogil=True)
    def update_B(B, Q, G, binary_iterations=20):
        """
        Updating the embedding matrix.
        """
        dimensions = B.shape[-1]
        dimensions_arr = np.arange(dimensions)
        for _ in range(binary_iterations):
            for d in range(dimensions):
                sel = dimensions_arr != d
                B[:, d] = np.sign(Q[:, d] - (B[:, sel] @ G[sel, :] @ G[:, np.array([d])])[:, 0])

        return B

    def get_embedding(self) -> np.array:
        """Getting the node embedding."""
        embedding = self._embedding
        return embedding
