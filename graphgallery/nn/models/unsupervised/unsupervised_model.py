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

from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from graphgallery.nn.models import BaseModel
from graphgallery import asintarr, Bunch


class UnsupervisedModel(BaseModel):
    """
        Base model for unsupervised learning.

        Arguments:
        ----------
            adj: shape (N, N), Scipy sparse matrix if  `is_adj_sparse=True`, 
                Numpy array-like (or matrix) if `is_adj_sparse=False`.
                The input `symmetric` adjacency matrix, where `N` is the number 
                of nodes in graph.
            x: shape (N, F), Scipy sparse matrix if `is_x_sparse=True`, 
                Numpy array-like (or matrix) if `is_x_sparse=False`.
                The input node feature matrix, where `F` is the dimension of features.
            labels: Numpy array-like with shape (N,)
                The ground-truth labels for all nodes in graph.
            device (String, optional): 
                The device where the model is running on. You can specified `CPU` or `GPU` 
                for the model. (default: :str: `CPU:0`, i.e., running on the 0-th `CPU`)
            seed (Positive integer, optional): 
                Used in combination with `tf.random.set_seed` & `np.random.seed` & `random.seed`  
                to create a reproducible sequence of tensors across multiple calls. 
                (default :obj: `None`, i.e., using random seed)
            name (String, optional): 
                Specified name for the model. (default: :str: `class.__name__`)

    """

    def __init__(self, adj, x, labels=None, device='CPU:0', seed=None, name=None, **kwargs):
        super().__init__(adj, x, labels, device, seed, name, **kwargs)

        self.embeddings = None
        self.classifier = LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='auto', random_state=seed)
        self.train_paras.update(Bunch(classifier=self.classifier))

    def build(self):
        raise NotImplementedError

    def get_embeddings(self):
        raise NotImplementedError

    def train(self, index):
        if not self.embeddings:
            self.get_embeddings()

        index = asintarr(index)
        self.classifier.fit(self.embeddings[index], self.labels[index])

    def predict(self, index):
        index = asintarr(index)
        logit = self.classifier.predict_proba(self.embeddings[index])
        return logit

    def test(self, index):
        index = asintarr(index)
        y_true = self.labels[index]
        y_pred = self.classifier.predict(self.embeddings[index])
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy

    @staticmethod
    def normalize_embedding(embeddings):
        return normalize(embeddings)
    
    def __repr__(self):
        return 'UnSupervised model: ' + self.name + ' in ' + self.device    
