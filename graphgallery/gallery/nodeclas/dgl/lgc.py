import torch
import graphgallery.nn.models.dgl as models
from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import NodeClasTrainer

from graphgallery.gallery.nodeclas import DGL


@DGL.register()
class LGC(NodeClasTrainer):
    """
        Implementation of Linear Graph Convolution (LGC). 
        `Simple Graph Convolutional Networks <https://arxiv.org/abs/2106.05809>`
        Pytorch implementation: <https://github.com/lpasa/LGC>

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
                   hids=[],
                   acts=[],
                   dropout=0.,
                   K=20,
                   bias=True):

        model = models.LGC(self.graph.num_feats,
                           self.graph.num_classes,
                           hids=hids,
                           acts=acts,
                           K=K,
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

    def config_optimizer(self) -> torch.optim.Optimizer:
        lr = self.cfg.get('lr', 0.2)
        weight_decay = self.cfg.get('weight_decay', 5e-5)
        return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)


@DGL.register()
class EGC(LGC):
    """
        Implementation of Exponential Graph Convolution (EGC). 
        `Simple Graph Convolutional Networks <https://arxiv.org/abs/2106.05809>`
        Pytorch implementation: <https://github.com/lpasa/LGC>

    """

    def model_step(self,
                   hids=[],
                   acts=[],
                   dropout=0.,
                   K=5,
                   bias=True):

        model = models.EGC(self.graph.num_feats,
                           self.graph.num_classes,
                           hids=hids,
                           acts=acts,
                           K=K,
                           dropout=dropout,
                           bias=bias)

        return model

    def config_optimizer(self) -> torch.optim.Optimizer:
        lr = self.cfg.get('lr', 0.2)
        weight_decay = self.cfg.get('weight_decay', 5e-4)
        return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)


@DGL.register()
class hLGC(NodeClasTrainer):
    """
        Implementation of Hyper Linear Graph Convolution (hLGC). 
        `Simple Graph Convolutional Networks <https://arxiv.org/abs/2106.05809>`
        Pytorch implementation: <https://github.com/lpasa/LGC>

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
                   hids=[],
                   acts=[],
                   dropout=0.,
                   K=10,
                   bias=True):

        model = models.hLGC(self.graph.num_feats,
                            self.graph.num_classes,
                            hids=hids,
                            acts=acts,
                            K=K,
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
