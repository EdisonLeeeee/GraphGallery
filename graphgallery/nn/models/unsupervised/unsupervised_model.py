from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import random
import datetime
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from graphgallery.nn.models import BaseModel


class UnsupervisedModel(BaseModel):
    """
        Base model for unsupervised learning.

        Arguments:
        ----------
            adj: `scipy.sparse.csr_matrix` (or `csc_matrix`) with shape (N, N)
                The input `symmetric` adjacency matrix, where `N` is the number of nodes 
                in graph.
            x: `np.array` with shape (N, F)
                The input node feature matrix, where `F` is the dimension of node features.
            labels: `np.array` with shape (N,)
                The ground-truth labels for all nodes in graph.
            device (String, optional): 
                The device where the model is running on. You can specified `CPU` or `GPU` 
                for the model. (default: :obj: `CPU:0`, i.e., the model is running on 
                the 0-th device `CPU`)
            seed (Positive integer, optional): 
                Used in combination with `tf.random.set_seed & np.random.seed & random.seed` 
                to create a reproducible sequence of tensors across multiple calls. 
                (default :obj: `None`, i.e., using random seed)
            name (String, optional): 
                Name for the model. (default: name of class)

    """

    def __init__(self, adj, x, labels=None, device='CPU:0', seed=None, name=None, **kwargs):
        super().__init__(adj, x, labels, device, seed, name, **kwargs)
        
        self.embeddings = None
        self.clssifier = LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='auto', random_state=seed)

    def build(self):
        raise NotImplementedError

    def get_embeddings(self):
        raise NotImplementedError

    def train(self, index):
        if self.embeddings is None:
            self.get_embeddings()

        index = self.to_int(index)
        self.clssifier.fit(self.embeddings[index], self.labels[index])

    def predict(self, index):
        index = self.to_int(index)
        logit = self.clssifier.predict_proba(self.embeddings[index])
        return logit

    def test(self, index):
        index = self.to_int(index)
        y_true = self.labels[index]
        y_pred = self.clssifier.predict(self.embeddings[index])
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy

    @staticmethod
    def normalize_embedding(embeddings):
        return normalize(embeddings)
