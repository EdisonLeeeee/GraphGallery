import torch
import graphgallery.nn.models.pyg as models
from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyG
from graphgallery.gallery import Trainer


@PyG.register()
class PDN(Trainer):
    """
        Implementation of Pathfinder Discovery Networks (PDN). 
        `Pathfinder Discovery Networks for Neural Message Passing 
        <https://arxiv.org/abs/2010.12878>`
        Pytorch implementation: <https://github.com/benedekrozemberczki/PDN>

    """

    def data_step(self,
                  edge_transform=None,
                  feat_transform=None,
                  edge_feat_transform=None):

        graph = self.graph
        edge_index, edge_weight = gf.get(edge_transform)(graph.edge_index, graph.edge_weight)
        attr_matrix = gf.get(feat_transform)(graph.attr_matrix)
        edge_attr = gf.get(edge_feat_transform)(graph.edge_attr)

        feat, edge_index, edge_feat = gf.astensors(attr_matrix,
                                                   edge_index,
                                                   edge_attr,
                                                   device=self.data_device)
        self.register_cache(feat=feat, edge_index=edge_index, edge_feat=edge_feat)

    def model_step(self,
                   hids=[32],
                   acts=['relu'],
                   pdn_hids=32,
                   dropout=0.5,
                   bias=True):

        model = models.PDN(self.graph.num_feats,
                           self.graph.num_classes,
                           self.graph.num_edge_feats,
                           hids=hids,
                           pdn_hids=pdn_hids,
                           acts=acts,
                           dropout=dropout,
                           bias=bias)

        return model

    def config_train_data(self, index):

        labels = self.graph.label[index]
        sequence = FullBatchSequence([self.cache.feat,
                                      self.cache.edge_index,
                                      self.cache.edge_feat],
                                     labels,
                                     out_index=index,
                                     device=self.data_device)
        return sequence

    def config_optimizer(self) -> torch.optim.Optimizer:
        lr = self.cfg.get('lr', 0.01)
        weight_decay = self.cfg.get('weight_decay', 5e-5)
        return torch.optim.Adam(self.model.parameters(),
                                weight_decay=weight_decay, lr=lr)
