import numpy as np
import graphgallery as gg
import graphgallery.functional as gf
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


class Trainer:
    def __init__(self, *, seed=None, name=None, **kwargs):
        kwargs.pop("__class__", None)
        self.set_hyparas(kwargs)

        gg.set_seed(seed)

        self.seed = seed
        self.name = name or self.__class__.__name__

        self._model = None
        self._graph = None
        self._cache = gf.BunchDict()
        self._embedding = None

    def register_cache(self, **kwargs):
        self._cache.update(kwargs)

    def fit(self, graph, *args, **kwargs):
        graph = getattr(graph, "adj_matrix", graph)
        self.fit_step(graph, *args, **kwargs)
        return self

    def fit_step(self, graph):
        raise NotImplementedError

    def get_embedding(self, normalize=True) -> np.ndarray:
        """Getting the node embedding."""
        embedding = self._embedding
        if normalize and embedding is not None:
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
        accuracy = metrics.accuracy_score(y_test, y_pred)
        macro_f1 = metrics.f1_score(y_test, y_pred, average='macro')
        micro_f1 = metrics.f1_score(y_test, y_pred, average='micro')

        return gf.BunchDict({'micro_f1': micro_f1.item(), 'macro_f1': macro_f1.item(), 'accuracy': accuracy.item()})

    def evaluate_linkpred(self, train_edges, y_train, test_edges, y_test):
        embedding = self.get_embedding()
        x_train = np.abs(embedding[train_edges[0]] - embedding[train_edges[1]])
        x_test = np.abs(embedding[test_edges[0]] - embedding[test_edges[1]])

        clf = LogisticRegression(solver="lbfgs",
                                 max_iter=1000,
                                 multi_class='auto',
                                 random_state=self.seed)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        y_pred_prob = clf.predict_proba(x_test)[:, 1]
        micro_f1 = metrics.f1_score(y_test, y_pred)
        ap = metrics.average_precision_score(y_test, y_pred_prob)
        auc = metrics.roc_auc_score(y_test, y_pred_prob)

        return gf.BunchDict({'micro_f1': micro_f1.item(), 'AUC': auc.item(), 'AP': ap.item()})

    def set_hyparas(self, kwargs: dict):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.hyparas = kwargs

    @property
    def graph(self):
        graph = self._graph
        if graph is None:
            raise KeyError("You haven't pass any graph instance.")
        return graph

    @graph.setter
    def graph(self, graph):
        assert graph is None or isinstance(graph, gg.data.BaseGraph)
        if graph is not None:
            self._graph = graph.copy()

    def empty_cache(self):
        self._cache = gf.BunchDict()
        import gc
        gc.collect()

    @property
    def cache(self):
        return self._cache

    def __repr__(self):
        para_str = ""
        for k, v in self.hyparas.items():
            para_str += f'{k}={v},\n'

        return f"{self.name}({para_str})"

    __str__ = __repr__
