import torch
import graphgallery.nn.models.pyg as models
from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyG
from graphgallery.gallery.nodeclas import NodeClasTrainer


@PyG.register()
class SGC(NodeClasTrainer):
    """
        Implementation of Simplifying Graph Convolutional Networks (SGC).
        `Simplifying Graph Convolutional Networks <https://arxiv.org/abs/1902.07153>`
        Pytorch implementation: <https://github.com/Tiiiger/SGC>

    """

    def data_step(self,
                  adj_transform=None,
                  feat_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(feat_transform)(graph.attr_matrix)

        feat, edges = gf.astensors(attr_matrix, adj_matrix, device=self.data_device)

        # ``edges`` and ``feat`` are cached for later use
        self.register_cache(feat=feat, edges=edges)

    def model_step(self,
                   hids=[],
                   acts=[],
                   dropout=0.,
                   bias=True,
                   K=2):

        model = models.SGC(self.graph.num_feats,
                           self.graph.num_classes,
                           K=K,
                           acts=acts,
                           dropout=dropout,
                           bias=bias)

        return model

    def config_train_data(self, index):

        labels = self.graph.label[index]
        sequence = FullBatchSequence([self.cache.feat, *self.cache.edges],
                                     labels,
                                     out_index=index,
                                     device=self.data_device)
        return sequence

    def config_optimizer(self) -> torch.optim.Optimizer:
        lr = self.cfg.get('lr', 0.2)
        weight_decay = self.cfg.get('weight_decay', 5e-5)
        return torch.optim.Adam(self.model.parameters(),
                                weight_decay=weight_decay, lr=lr)
