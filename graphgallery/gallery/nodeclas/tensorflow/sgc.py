import tensorflow as tf

from graphgallery.data.sequence import FullBatchSequence
from graphgallery.nn.layers.tensorflow import SGConv
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import TensorFlow
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model


@TensorFlow.register()
class SGC(Trainer):
    """
        Implementation of Simplifying Graph Convolutional Networks (SGC). 
        `Simplifying Graph Convolutional Networks <https://arxiv.org/abs/1902.07153>`
        Pytorch implementation: <https://github.com/Tiiiger/SGC>

    """

    def data_step(self,
                  adj_transform="normalize_adj",
                  attr_transform=None,
                  K=2):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        node_attr = gf.get(attr_transform)(graph.node_attr)

        X, A = gf.astensors(node_attr, adj_matrix, device=self.data_device)

        # To avoid this tensorflow error in large dataset:
        # InvalidArgumentError: Cannot use GPU when output.shape[1] * nnz(a) > 2^31 [Op:SparseTensorDenseMatMul]
        if X.shape[1] * adj_matrix.nnz > 2**31:
            device = "CPU"
        else:
            device = self.device

        with tf.device(device):
            X = SGConv(K=K)([X, A])

        with tf.device(self.device):
            # ``A`` and ``X`` are cached for later use
            self.register_cache(X=X, A=A)

    def model_step(self,
                   hids=[],
                   acts=[],
                   dropout=0.5,
                   weight_decay=5e-5,
                   lr=0.2,
                   bias=True):

        model = get_model("MLP", self.backend)
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
        labels = self.graph.node_label[index]
        index = gf.astensor(index)

        X = tf.gather(self.cache.X, index)
        sequence = FullBatchSequence(X, labels, device=self.data_device)
        return sequence
