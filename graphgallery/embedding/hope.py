import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn import preprocessing

class HOPE:
    r"""An implementation of `"HOPE" <https://www.kdd.org/kdd2016/papers/files/rfp0184-ouA.pdf>`_
    from the KDD '16 paper "Asymmetric Transitivity Preserving Graph Embedding". The procedure uses
    sparse SVD on the neighbourhood overlap matrix. The singular value rescaled left and right 
    singular vectors are used as the node embeddings after concatenation.
    """
    def __init__(self, dimensions: int=128, seed: int=None):

        self.dimensions = dimensions
        self.seed = seed

    def rescaled_decomposition(self, S):
        """
        Decomposing the similarity matrix.
        """
        U, sigmas, Vt = sp.linalg.svds(S, k=int(self.dimensions/2))
        sigmas = np.diagflat(np.sqrt(sigmas))
        self._left_embedding = np.dot(U, sigmas)
        self._right_embedding = np.dot(Vt.T, sigmas)

    def fit(self, graph: sp.csr_matrix):
        """
        Fitting a HOPE model.
        """
        S = graph @ graph
        self.rescaled_decomposition(S)

    def get_embedding(self, normalize=False) -> np.array:
        r"""Getting the node embedding.
        """
        embedding = np.concatenate([self._left_embedding, self._right_embedding], axis=1)
        if normalize:
            embedding = preprocessing.normalize(embedding)            
        return embedding
        
