import numpy as np

from graphgallery.sequence import SAGESequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import TensorFlow
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model


@TensorFlow.register()
class GraphSAGE(Trainer):
    """
        Implementation of SAmple and aggreGatE Graph Convolutional Networks (GraphSAGE). 
        `Inductive Representation Learning on Large Graphs <https://arxiv.org/abs/1706.02216>`
        Tensorflow 1.x implementation: <https://github.com/williamleif/GraphSAGE>
        Pytorch implementation: <https://github.com/williamleif/graphsage-simple/>
    """

    def custom_setup(self,
                     batch_size_train=512,
                     batch_size_test=20000):

        self.cfg.fit.batch_size = batch_size_train
        self.cfg.evaluate.batch_size = batch_size_test

    def data_step(self,
                  adj_transform=None,
                  attr_transform=None,
                  sizes=[15, 5]):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        node_attr = gf.get(attr_transform)(graph.node_attr)

        X, A = gf.astensors(node_attr, device=self.data_device), adj_matrix

        # ``A`` and ``X`` are cached for later use
        self.register_cache(X=X, A=A)

    def model_step(self,
                   hids=[32],
                   acts=['relu'],
                   dropout=0.5,
                   weight_decay=5e-4,
                   lr=0.01,
                   bias=True,
                   output_normalize=False,
                   aggregator='mean'):

        model = get_model("GraphSAGE", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      hids=hids,
                      acts=acts,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      bias=bias,
                      aggregator=aggregator,
                      output_normalize=output_normalize,
                      sizes=self.cfg.data.sizes)

        return model

    def train_loader(self, index):
        labels = self.graph.node_label[index]
        sequence = SAGESequence(
            inputs=[self.cache.X, self.cache.A],
            nodes=index,
            y=labels,
            shuffle=True,
            batch_size=self.cfg.fit.batch_size,
            sizes=self.cfg.data.sizes,
            device=self.data_device)
        return sequence

    def test_loader(self, index):
        labels = self.graph.node_label[index]
        sequence = SAGESequence(
            inputs=[self.cache.X, self.cache.A],
            nodes=index,
            y=labels,
            batch_size=self.cfg.evaluate.batch_size,
            sizes=self.cfg.data.sizes,
            device=self.data_device)
        return sequence
