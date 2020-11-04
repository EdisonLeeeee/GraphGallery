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

from graphgallery.nn.gallery import BaseModel
from graphgallery.functional import asintarr


class UnsupervisedModel(BaseModel):
    """Base model for unsupervised learning.

    """

    def __init__(self, *graph, device='cpu:0', seed=None, name=None, **kwargs):
        """Create an unsupervised model.

        Parameters:
        ----------
        graph: An instance of `graphgallery.data.Graph` or a tuple (list) of inputs.
            A sparse, attributed, labeled graph.
        device: string. optional 
            The device where the model is running on. You can specified `CPU` or `GPU` 
            for the model. (default: :str: `CPU:0`, i.e., running on the 0-th `CPU`)
        seed: interger scalar. optional 
            Used in combination with `tf.random.set_seed` & `np.random.seed` 
            & `random.seed` to create a reproducible sequence of tensors across 
            multiple calls. (default :obj: `None`, i.e., using random seed)
        name: string. optional
            Specified name for the model. (default: :str: `class.__name__`)        

        """
        super().__init__(*graph, device=device, seed=seed, name=name, **kwargs)

        self.embeddings = None
        self.classifier = LogisticRegression(solver='lbfgs',
                                             max_iter=1000,
                                             multi_class='auto',
                                             random_state=seed)

    def build(self):
        raise NotImplementedError

    def get_embeddings(self):
        raise NotImplementedError

    def train(self, index):
        if not self.embeddings:
            self.get_embeddings()

        index = asintarr(index)
        self.classifier.fit(self.embeddings[index], self.graph.labels[index])

    def predict(self, index):
        index = asintarr(index)
        logit = self.classifier.predict_proba(self.embeddings[index])
        return logit

    def test(self, index):
        index = asintarr(index)
        y_true = self.graph.labels[index]
        y_pred = self.classifier.predict(self.embeddings[index])
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy

    @staticmethod
    def normalize_embedding(embeddings):
        return normalize(embeddings)
