import torch
import graphgallery.nn.models.dgl as models
from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import NodeClasTrainer
from graphgallery.gallery.nodeclas import DGL


@DGL.register()
class GAT(NodeClasTrainer):
    """
        Implementation of Graph Attention Networks (GAT).
        `Graph Attention Networks <https://arxiv.org/abs/1710.10903>`
        Tensorflow 1.x implementation: <https://github.com/PetarV-/GAT>
        Pytorch implementation: <https://github.com/Diego999/pyGAT>
        Keras implementation: <https://github.com/danielegrattarola/keras-gat>

    """

    def data_step(self,
                  adj_transform="add_self_loop",
                  feat_transform=None):
        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(feat_transform)(graph.attr_matrix)
        feat, g = gf.astensors(attr_matrix, adj_matrix, device=self.data_device)

        # ``g`` and ``feat`` are cached for later use
        self.register_cache(feat=feat, g=g)

    def model_step(self,
                   hids=[8],
                   num_heads=[8],
                   acts=['elu'],
                   dropout=0.6,
                   bias=True,
                   include=["num_heads"]):

        model = models.GAT(self.graph.num_feats,
                           self.graph.num_classes,
                           hids=hids,
                           num_heads=num_heads,
                           acts=acts,
                           dropout=dropout,
                           bias=bias)

        return model

    def config_train_data(self, index):
        labels = self.graph.label[index]
        sequence = FullBatchSequence(inputs=[self.cache.feat, self.cache.g],
                                     y=labels,
                                     out_index=index,
                                     device=self.data_device,
                                     escape=type(self.cache.g))
        return sequence
