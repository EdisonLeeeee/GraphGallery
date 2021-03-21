import numpy as np

from graphgallery.sequence import SAGEMiniBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import TensorFlow
from graphgallery.gallery.nodeclas import Trainer
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
                     num_samples_train=[15, 5],
                     num_samples_test=[15, 5]):
        self.cfg.fit.num_samples = num_samples_train
        self.cfg.evaluate.num_samples = num_samples_test

    def data_step(self,
                  adj_transform="neighbor_sampler",
                  attr_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        node_attr = gf.get(attr_transform)(graph.node_attr)
       # pad with a dummy zero vector
        node_attr = np.vstack([node_attr, np.zeros(node_attr.shape[1],
                                                   dtype=self.floatx)])

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
                   aggregator='mean',
                   num_samples=[15, 5]):

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
                      num_samples=num_samples)

        return model

    def train_loader(self, index):

        labels = self.graph.node_label[index]
        sequence = SAGEMiniBatchSequence(
            [self.cache.X, self.cache.A, index],
            labels,
            num_samples=self.cfg.model.num_samples,
            device=self.data_device)
        return sequence
