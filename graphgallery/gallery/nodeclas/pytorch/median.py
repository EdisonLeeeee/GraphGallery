import torch
import graphgallery.nn.models.pytorch as models
from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from graphgallery.gallery.nodeclas import NodeClasTrainer


@PyTorch.register()
class MedianGCN(NodeClasTrainer):
    """
        Implementation of Graph Convolutional Networks with Median aggregation (MedianGCN). 
        `Understanding Structural Vulnerability in Graph Convolutional Networks 
        <https://arxiv.org/abs/2108.06280>`
        Pytorch implementation: <https://github.com/EdisonLeeeee/MedianGCN>

    """

    def data_step(self,
                  adj_transform="add_self_loop",
                  feat_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(feat_transform)(graph.attr_matrix)

        feat, adj_lists = gf.astensors(attr_matrix, adj_matrix.tolil().rows, device=self.data_device)

        # ``adj_lists`` and ``feat`` are cached for later use
        self.register_cache(feat=feat, adj_lists=adj_lists)

    def model_step(self,
                   hids=[16],
                   acts=['relu'],
                   dropout=0.5,
                   bias=False):

        model = models.MedianGCN(self.graph.num_feats,
                                 self.graph.num_classes,
                                 hids=hids,
                                 acts=acts,
                                 dropout=dropout,
                                 bias=bias)
        return model

    def config_train_data(self, index):

        labels = self.graph.label[index]
        sequence = FullBatchSequence([self.cache.feat, self.cache.adj_lists],
                                     labels,
                                     out_index=index,
                                     device=self.data_device)
        return sequence

    def config_optimizer(self) -> torch.optim.Optimizer:
        lr = self.cfg.get('lr', 0.01)
        weight_decay = self.cfg.get('weight_decay', 1e-4)
        model = self.model
        return torch.optim.Adam([dict(params=model.reg_paras,
                                      weight_decay=weight_decay),
                                 dict(params=model.non_reg_paras,
                                      weight_decay=0.)], lr=lr)


@PyTorch.register()
class TrimmedGCN(NodeClasTrainer):
    """
        Implementation of Graph Convolutional Networks with Trimmed mean aggregation (TrimmedGCN). 
        `Understanding Structural Vulnerability in Graph Convolutional Networks 
        <https://arxiv.org/abs/2108.06280>`
        Pytorch implementation: <https://github.com/EdisonLeeeee/MedianGCN>

    """

    def data_step(self,
                  adj_transform="add_self_loop",
                  feat_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(feat_transform)(graph.attr_matrix)

        feat, adj_lists = gf.astensors(attr_matrix, adj_matrix.tolil().rows, device=self.data_device)

        # ``adj_lists`` and ``feat`` are cached for later use
        self.register_cache(feat=feat, adj_lists=adj_lists)

    def model_step(self,
                   hids=[16],
                   acts=['relu'],
                   dropout=0.5,
                   alpha=0.45,
                   bias=False):

        model = models.TrimmedGCN(self.graph.num_feats,
                                  self.graph.num_classes,
                                  hids=hids,
                                  acts=acts,
                                  alpha=alpha,
                                  dropout=dropout,
                                  bias=bias)
        return model

    def config_train_data(self, index):

        labels = self.graph.label[index]
        sequence = FullBatchSequence([self.cache.feat, self.cache.adj_lists],
                                     labels,
                                     out_index=index,
                                     device=self.data_device)
        return sequence

    def config_optimizer(self) -> torch.optim.Optimizer:
        lr = self.cfg.get('lr', 0.01)
        weight_decay = self.cfg.get('weight_decay', 1e-4)
        model = self.model
        return torch.optim.Adam([dict(params=model.reg_paras,
                                      weight_decay=weight_decay),
                                 dict(params=model.non_reg_paras,
                                      weight_decay=0.)], lr=lr)
