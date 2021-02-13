import tensorflow as tf

from graphgallery.sequence import FullBatchSequence
from graphgallery.gallery.nodeclas import Trainer
from graphgallery.nn.models import get_model
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import TensorFlow


@TensorFlow.register()
class ChebyNet(Trainer):
    """
        Implementation of Chebyshev Graph Convolutional Networks (ChebyNet).
        `Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering <https://arxiv.org/abs/1606.09375>`
        Tensorflow 1.x implementation: <https://github.com/mdeff/cnn_graph>, <https://github.com/tkipf/gcn>
        Keras implementation: <https://github.com/aclyde11/ChebyGCN>

        This can be instantiated in the following way:

            trainer = ChebyNet(graph)
                with a `graphgallery.data.Graph` instance representing
                A sparse, attributed, labeled graph.

        """

    def process_step(self,
                     adj_transform="cheby_basis",
                     attr_transform=None,
                     graph_transform=None):

        graph = gf.get(graph_transform)(self.graph)
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
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

        model = get_model("ChebyNet", self.backend)
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
        sequence = FullBatchSequence([self.cache.X, *self.cache.A],
                                     labels,
                                     out_weight=index,
                                     device=self.device)
        return sequence
