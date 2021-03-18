from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

from graphgallery.gallery.nodeclas import Trainer
from graphgallery import functional as gf
import graphgallery as gg


class SklearnModel(Trainer):
    """Sklean based model for unsupervised learning."""

    def setup_cfg(self):
        self.backend = None
        cfg = gg.CfgNode()
        cfg.name = self.name
        cfg.seed = self.seed
        cfg.device = str(self.device)
        cfg.task = "Embedding"
        cfg.intx = self.intx
        cfg.floatx = self.floatx
        cfg.boolx = self.boolx
        cfg.backend = self.backend
        cfg.normalize_embedding = True

        cfg.build = gg.CfgNode()
        cfg.process = gg.CfgNode()

        cfg.classifier = gg.CfgNode()
        cfg.classifier.name = "LogisticRegression"
        cfg.classifier.solver = "lbfgs"
        cfg.classifier.max_iter = 1000
        cfg.classifier.multi_class = "auto"
        cfg.classifier.random_state = None

        self.cfg = cfg

    def build(self, **kwargs):
        self.model, kwargs = gf.wrapper(self.model_builder)(**kwargs)
        self.cfg.build.merge_from_dict(kwargs)
        self.classifier = self.classifier_builder()
        return self

    def builder(self, *args, **kwargs):
        raise NotImplementedError

    def classifier_builder(self):
        raise NotImplementedError

    def train_sequence(self, index, **kwargs):
        index = gf.asarray(index)
        return self.embeddings[index], self.graph.node_label[index]

    def train(self, index):
        x, y = self.train_sequence(index)
        self.classifier.fit(x, y)
        return gf.BunchDict(loss=None, accuracy=None)

    def predict(self, index):
        x, y = self.predict_sequence(index)
        logit = self.classifier.predict_proba(x)
        return logit

    def test(self, index):
        x, y = self.test_sequence(index)
        y_pred = self.classifier.predict(x)
        accuracy = accuracy_score(y, y_pred)
        return gf.BunchDict(loss=None, accuracy=accuracy)

    @staticmethod
    def normalize_embedding(embeddings):
        return normalize(embeddings)
