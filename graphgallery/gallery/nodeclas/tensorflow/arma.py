from graphgallery.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import TensorFlow
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model


@TensorFlow.register()
class ARMA(Trainer):
    """
        Implementation of ARMA model.
        `Graph Neural Networks with convolutional ARMA filters
        <https://arxiv.org/abs/1901.01343>`
        Tensorflow 2.x implementation: <https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/convolutional/arma_conv.py>
    """

    def data_step(self,
                  adj_transform="normalize_adj",
                  attr_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        node_attr = gf.get(attr_transform)(graph.node_attr)

        X, A = gf.astensors(node_attr, adj_matrix, device=self.data_device)

        # ``A`` and ``X`` are cached for later use
        self.register_cache(X=X, A=A)

    def model_step(self,
                   hids=[16],
                   acts=['elu'],
                   order=2,
                   iterations=1,
                   dropout=0.5,
                   weight_decay=5e-5,
                   lr=0.01,
                   bias=True):

        model = get_model("ARMA", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      hids=hids,
                      acts=acts,
                      order=order,
                      iterations=iterations,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      bias=bias)

        return model

    def train_loader(self, index):

        labels = self.graph.node_label[index]
        sequence = FullBatchSequence(inputs=[self.cache.X, self.cache.A],
                                     y=labels,
                                     out_index=index,
                                     device=self.data_device)
        return sequence
