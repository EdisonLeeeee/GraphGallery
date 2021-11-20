import torch
import graphgallery.nn.models.pyg as models
from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyG
from graphgallery.gallery.nodeclas import NodeClasTrainer


@PyG.register()
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

        feat = gf.astensor(attr_matrix, device=self.data_device)
        # without considering `edge_weight`
        edges = gf.astensor(adj_matrix, device=self.data_device)[0]

        # ``edges`` and ``feat`` are cached for later use
        self.register_cache(feat=feat, edges=edges)

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
        sequence = FullBatchSequence([self.cache.feat, self.cache.edges],
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
