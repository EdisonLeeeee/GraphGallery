import tensorflow as tf

from graphgallery.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery import TensorFlow
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model


@TensorFlow.register()
class DenseGCN(Trainer):
    """
        Implementation of Dense version of Graph Convolutional Networks (GCN).
        `[`Semi-Supervised Classification with Graph Convolutional Networks <https://arxiv.org/abs/1609.02907>`
        Tensorflow 1.x `Sparse version` implementation: <https://github.com/tkipf/gcn>
        Pytorch `Sparse version` implementation: <https://github.com/tkipf/pygcn>

    """

    def process_step(self,
                     adj_transform="normalize_adj",
                     attr_transform=None,
                     graph_transform=None):

        graph = gf.get(graph_transform)(self.graph)
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix).toarray()
        node_attr = gf.get(attr_transform)(graph.node_attr)

        X, A = gf.astensors(node_attr, adj_matrix, device=self.device)

        # ``A`` and ``X`` are cached for later use
        self.register_cache(X=X, A=A)

    def builder(self,
                hids=[16],
                acts=['relu'],
                dropout=0.5,
                weight_decay=5e-4,
                lr=0.01,
                bias=False,
                use_tfn=True):

        model = get_model("DenseGCN", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      hids=hids,
                      acts=acts,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      bias=bias)
        if use_tfn:
            model.use_tfn()

        return model

    def train_sequence(self, index):

        labels = self.graph.node_label[index]
        sequence = FullBatchSequence([self.cache.X, self.cache.A],
                                     labels,
                                     out_weight=index,
                                     device=self.device)
        return sequence
