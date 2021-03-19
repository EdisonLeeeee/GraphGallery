import numpy as np
import scipy.sparse as sp
from sklearn import preprocessing
from utils import normalized_laplacian_matrix


class LaplacianEigenmaps:
    r"""An implementation of `"Laplacian Eigenmaps" <https://papers.nips.cc/paper/1961-laplacian-eigenmaps-and-spectral-techniques-for-embedding-and-clustering>`_
    from the NIPS '01 paper "Laplacian Eigenmaps and Spectral Techniques for Embedding and Clustering".
    The procedure extracts the eigenvectors corresponding to the largest eigenvalues 
    of the graph Laplacian. These vectors are used as the node embedding.
    """

    def __init__(self, dimensions: int = 32, seed: int = None):

        self.dimensions = dimensions
        self.seed = seed

    def fit(self, graph: sp.csr_matrix):
        """
        Fitting a Laplacian EigenMaps model.
        """
        L_tilde = normalized_laplacian_matrix(graph)
        _, self._embedding = sp.linalg.eigsh(L_tilde, k=self.dimensions,
                                             which='SM', return_eigenvectors=True)

    def get_embedding(self, normalize=True) -> np.array:
        """Getting the node embedding."""
        embedding = self._embedding
        if normalize:
            embedding = preprocessing.normalize(embedding)
        return embedding
