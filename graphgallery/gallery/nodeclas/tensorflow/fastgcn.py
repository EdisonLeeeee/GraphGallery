import tensorflow as tf

from graphgallery.sequence import FastGCNBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import TensorFlow
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model


@TensorFlow.register()
class FastGCN(Trainer):
    """
        Implementation of Fast Graph Convolutional Networks (FastGCN).
        `FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling 
        <https://arxiv.org/abs/1801.10247>`
        Tensorflow 1.x implementation: <https://github.com/matenure/FastGCN>

    """

    def custom_setup(self):
        cfg = self.cfg.fit
        cfg.batch_size = 256
        cfg.rank = 100

        cfg = self.cfg.evaluate
        cfg.batch_size = None
        cfg.rank = None

    def data_step(self,
                  adj_transform="normalize_adj",
                  attr_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        node_attr = gf.get(attr_transform)(graph.node_attr)
        node_attr = adj_matrix @ node_attr

        X, A = gf.astensor(node_attr, device=self.data_device), adj_matrix

        # ``A`` and ``X`` are cached for later use
        self.register_cache(X=X, A=A)

    def model_step(self,
                   hids=[32],
                   acts=['relu'],
                   dropout=0.5,
                   weight_decay=5e-4,
                   lr=0.01,
                   bias=False):

        model = get_model("FastGCN", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      hids=hids,
                      acts=acts,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      bias=bias)

        return model

    def train_loader(self, index):
        cfg = self.cfg.fit

        labels = self.graph.node_label[index]
        adj_matrix = self.graph.adj_matrix[index][:, index]
        adj_matrix = self.transform.adj_transform(adj_matrix)

        X = tf.gather(self.cache.X, index)
        sequence = FastGCNBatchSequence([X, adj_matrix],
                                        labels,
                                        batch_size=cfg.batch_size,
                                        rank=cfg.rank,
                                        device=self.data_device)
        return sequence

    def test_loader(self, index):
        cfg = self.cfg.evaluate

        labels = self.graph.node_label[index]
        A = self.cache.A[index]

        sequence = FastGCNBatchSequence([self.cache.X, A],
                                        labels,
                                        batch_size=cfg.batch_size,
                                        rank=cfg.rank,
                                        device=self.data_device)
        return sequence
