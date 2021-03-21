import tensorflow as tf

from graphgallery.sequence import FullBatchSequence

from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import TensorFlow
from graphgallery.nn.models import get_model
from ..gcn import GCN


@TensorFlow.register()
class GCNA(GCN):
    """
    GCN + node attribute matrix

    Implementation of Graph Convolutional Networks(GCN) concated with node attribute matrix.
    `Semi - Supervised Classification with Graph Convolutional Networks 
    <https://arxiv.org/abs/1609.02907>`
    GCN Tensorflow 1.x implementation: <https://github.com/tkipf/gcn>
    GCN Pytorch implementation: <https://github.com/tkipf/pygcn>

    """

    def model_step(self,
                   hids=[16],
                   acts=['relu'],
                   dropout=0.5,
                   weight_decay=5e-4,
                   lr=0.01,
                   bias=False,
                   use_tfn=True):

        model = get_model("GCNA", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      hids=hids,
                      acts=acts,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      bias=bias)

        return model
