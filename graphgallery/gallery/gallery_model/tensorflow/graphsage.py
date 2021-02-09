import numpy as np
import tensorflow as tf

from graphgallery.sequence import SAGEMiniBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery import TensorFlow
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
                     num_samples_train=[15, 5],
                     num_samples_test=[15, 5]):
        self.cfg.train.num_samples = num_samples_train
        self.cfg.test.num_samples = num_samples_test

    def process_step(self,
                     adj_transform="neighbor_sampler",
                     attr_transform=None,
                     graph_transform=None):

        graph = gf.get(graph_transform)(self.graph)
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        node_attr = gf.get(attr_transform)(graph.node_attr)
       # pad with a dummy zero vector
        node_attr = np.vstack([node_attr, np.zeros(node_attr.shape[1],
                                                   dtype=self.floatx)])

        X, A = gf.astensors(node_attr, device=self.device), adj_matrix

        # ``A`` and ``X`` are cached for later use
        self.register_cache(X=X, A=A)

    def builder(self,
                hids=[32],
                acts=['relu'],
                dropout=0.5,
                weight_decay=5e-4,
                lr=0.01,
                bias=True,
                output_normalize=False,
                aggregator='mean',
                num_samples=[15, 5],
                use_tfn=True):

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
        if use_tfn:
            model.use_tfn()

        return model

    def train_sequence(self, index):

        labels = self.graph.node_label[index]
        sequence = SAGEMiniBatchSequence(
            [self.cache.X, self.cache.A, index],
            labels,
            num_samples=self.cfg.model.num_samples,
            device=self.device)
        return sequence
