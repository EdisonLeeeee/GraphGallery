from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from graphgallery.gallery import GraphModel
from graphgallery.functional import asarray, BunchDict


class SklearnModel(GraphModel):
    """Sklean based model for unsupervised learning."""

    def __init__(self, graph, device='cpu', seed=None, name=None, **kwargs):
        r"""Create an Sklean based model 
        Parameters:
        ----------
        graph: An instance of graphgallery Graph.
            A sparse, attributed, labeled graph.
        device: string. optional 
            The device where the model is running on. You can specified `CPU` or `GPU` 
            for the model. (default: :str: `cpu`, i.e., running on the 0-th `CPU`)
        seed: interger scalar. optional 
            Used in combination with `tf.random.set_seed` & `np.random.seed` 
            & `random.seed` to create a reproducible sequence of tensors across 
            multiple calls. (default :obj: `None`, i.e., using random seed)
        name: string. optional
            Specified name for the model. (default: :str: `class.__name__`)        

        """
        super().__init__(graph, device=device, seed=seed, name=name, **kwargs)

    def build(self):
        self._embeddings = None
        self.classifier = LogisticRegression(solver='lbfgs',
                                             max_iter=1000,
                                             multi_class='auto',
                                             random_state=self.seed)

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = self.get_embeddings()
        return self._embeddings

    def train(self, index):
        index = asarray(index)
        self.classifier.fit(self.embeddings[index],
                            self.graph.node_label[index])

    def predict(self, index):
        index = asarray(index)
        logit = self.classifier.predict_proba(self.embeddings[index])
        return logit

    def test(self, index):
        index = asarray(index)
        y_true = self.graph.node_label[index]
        y_pred = self.classifier.predict(self.embeddings[index])
        accuracy = accuracy_score(y_true, y_pred)
        return BunchDict(loss=None, accuracy=accuracy)

    @staticmethod
    def normalize_embedding(embeddings):
        return normalize(embeddings)
