import torch
import graphgallery.nn.models.pyg as models
from graphgallery.data.sequence import PyGSAGESequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyG
from graphgallery.gallery import Trainer


@PyG.register()
class GraphSAGE(Trainer):
    """
        Implementation of SAmple and aggreGatE Graph Convolutional Networks (GraphSAGE). 
        `Inductive Representation Learning on Large Graphs <https://arxiv.org/abs/1706.02216>`
        Tensorflow 1.x implementation: <https://github.com/williamleif/GraphSAGE>
        Pytorch implementation: <https://github.com/williamleif/graphsage-simple/>
    """

    def data_step(self,
                  adj_transform=None,
                  feat_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(feat_transform)(graph.attr_matrix)

        X, A = gf.astensors(attr_matrix, device=self.data_device), adj_matrix

        # ``A`` and ``X`` are cached for later use
        self.register_cache(X=X, A=A)

    def model_step(self,
                   hids=[32],
                   acts=['relu'],
                   dropout=0.5,
                   bias=False,
                   output_normalize=False):

        model = models.GraphSAGE(self.graph.num_feats,
                                 self.graph.num_classes,
                                 hids=hids,
                                 acts=acts,
                                 dropout=dropout,
                                 bias=bias,
                                 output_normalize=output_normalize)
        return model

    def config_train_data(self, index):
        labels = self.graph.label[index]
        batch_size_train = self.cfg.get("batch_size_train", 512)
        sizes = self.cfg.get("sizes", [15, 5])

        sequence = PyGSAGESequence(
            inputs=[self.cache.X, self.cache.A],
            nodes=index,
            y=labels,
            shuffle=True,
            batch_size=batch_size_train,
            sizes=sizes,
            device=self.data_device)
        return sequence

    def config_test_data(self, index):
        labels = self.graph.label[index]
        batch_size_test = self.cfg.get("batch_size_test", 20000)
        sizes = self.cfg.get("sizes", [15, 5])

        sequence = PyGSAGESequence(
            inputs=[self.cache.X, self.cache.A],
            nodes=index,
            y=labels,
            batch_size=batch_size_test,
            sizes=sizes,
            device=self.data_device)
        return sequence
