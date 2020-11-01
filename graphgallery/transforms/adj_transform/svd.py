import numpy as np
import scipy.sparse as sp
from graphgallery.transforms import Transform
from graphgallery.utils.shape import repeat
from graphgallery.utils.decorators import MultiInputs


class SVD(Transform):

    def __init__(self, k=50, threshold=0.01, binaryzation=False):
        super().__init__()
        self.k = k
        self.threshold = threshold
        self.binaryzation = binaryzation

    def __call__(self, adj_matrix):
        return svd(adj_matrix, k=self.k,
                   threshold=self.threshold,
                   binaryzation=self.binaryzation)

    def __repr__(self):
        return f"{self.__class__.__name__}(k={self.k}, threshold={self.threshold}, binaryzation={self.binaryzation})"


@MultiInputs()
def svd(adj_matrix, k=50, threshold=0.01, binaryzation=False):
    
    adj_matrix = adj_matrix.asfptype()
    U, S, V = sp.linalg.svds(adj_matrix, k=k)
    adj_matrix = (U*S) @ V

    if threshold is not None:
        # sparsification
        adj_matrix[adj_matrix <= threshold] = 0.

    adj_matrix = sp.csr_matrix(adj_matrix)

    if binaryzation:
        # TODO
        adj_matrix.data[adj_matrix.data > 0] = 1.0  
        
    return adj_matrix
