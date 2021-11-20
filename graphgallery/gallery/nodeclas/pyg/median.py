import torch
import graphgallery.nn.models.pyg as models
from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyG
from graphgallery.gallery.nodeclas import NodeClasTrainer


@PyG.register()
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

        feat, edges = gf.astensors(attr_matrix, adj_matrix, device=self.data_device)

        # ``edges`` and ``feat`` are cached for later use
        self.register_cache(feat=feat, edges=edges)

    def model_step(self,
                   hids=[16],
                   acts=['relu'],
                   dropout=0.5,
                   bias=True):

        model = models.MedianGCN(self.graph.num_feats,
                                 self.graph.num_classes,
                                 hids=hids,
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
        lr = self.cfg.get('lr', 0.01)
        weight_decay = self.cfg.get('weight_decay', 5e-4)
        model = self.model
        return torch.optim.Adam([dict(params=model.reg_paras,
                                      weight_decay=weight_decay),
                                 dict(params=model.non_reg_paras,
                                      weight_decay=0.)], lr=lr)
