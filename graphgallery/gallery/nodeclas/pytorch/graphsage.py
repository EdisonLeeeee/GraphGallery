import numpy as np
from graphgallery.sequence import SAGESequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model


@PyTorch.register()
class GraphSAGE(Trainer):
    """
        Implementation of SAmple and aggreGatE Graph Convolutional Networks (GraphSAGE). 
        `Inductive Representation Learning on Large Graphs <https://arxiv.org/abs/1706.02216>`
        Tensorflow 1.x implementation: <https://github.com/williamleif/GraphSAGE>
        Pytorch implementation: <https://github.com/williamleif/graphsage-simple/>
    """

    def data_step(self,
                  adj_transform=None,
                  attr_transform=None):

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
                   bias=False,
                   output_normalize=False,
                   aggregator='mean',
                   sizes=[15, 5]):

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
                      sizes=sizes)
        return model

    def train_loader(self, index):
        labels = self.graph.node_label[index]
        sequence = SAGESequence(
            inputs=[self.cache.X, self.cache.A],
            nodes=index,
            y=labels,
            batch_size=256,
            sizes=self.cfg.model.sizes,
            device=self.data_device)
        return sequence
