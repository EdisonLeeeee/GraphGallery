import numpy as np
import scipy.sparse as sp
from sklearn import preprocessing
from graphgallery.gallery import Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from .default import default_cfg_setup


class Trainer(Model):
    def __init__(self, *, seed=None, name=None, **kwargs):
        super().__init__(seed=seed, name=name, **kwargs)
        self._embedding = None

    def setup_cfg(self):
        default_cfg_setup(self.cfg)

    def fit(self, graph):
        self.fit_step(graph)

    def get_embedding(self, normalize=True) -> np.array:
        """Getting the node embedding."""
        embedding = self._embedding
        if normalize:
            embedding = preprocessing.normalize(embedding)
        return embedding

    def evaluate_nodeclas(self, y, train_nodes, test_nodes):
        embedding = self.get_embedding()
        x_train = embedding[train_nodes]
        x_test = embedding[test_nodes]
        y_train = y[train_nodes]
        y_test = y[test_nodes]

        clf = LogisticRegression(solver="lbfgs",
                                 max_iter=1000,
                                 multi_class='auto',
                                 random_state=self.seed)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
