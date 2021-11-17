import torch
import graphgallery.nn.models.pytorch as models
from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from graphgallery.gallery import Trainer


@PyTorch.register()
class APPNP(Trainer):
    """Implementation of approximated personalized propagation of neural 
        predictions (APPNP).
        `Predict then Propagate: Graph Neural Networks meet Personalized
        PageRank" <https://arxiv.org/abs/1810.05997>`
        Tensorflow 1.x implementation: <https://github.com/klicperajo/ppnp>
        Pytorch implementation: <https://github.com/klicperajo/ppnp>
    """

    def data_step(self,
                  adj_transform="normalize_adj",
                  attr_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(attr_transform)(graph.attr_matrix)

        feat, adj = gf.astensors(attr_matrix, adj_matrix, device=self.data_device)

        # ``adj`` and ``feat`` are cached for later use
        self.register_cache(feat=feat, adj=adj)

    def model_step(self,
                   hids=[64],
                   acts=['relu'],
                   alpha=0.1,
                   K=10,
                   ppr_dropout=0.,
                   dropout=0.5,
                   bias=True):

        model = models.APPNP(self.graph.num_feats,
                             self.graph.num_classes,
                             hids=hids,
                             acts=acts,
                             alpha=alpha,
                             K=K,
                             ppr_dropout=ppr_dropout,
                             dropout=dropout,
                             bias=bias,
                             approximated=True)

        return model

    def config_train_data(self, index):

        labels = self.graph.label[index]
        sequence = FullBatchSequence(inputs=[self.cache.feat, self.cache.adj],
                                     y=labels,
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


@PyTorch.register()
class PPNP(Trainer):
    """Implementation of exact personalized propagation of neural 
        predictions (PPNP).
        `Predict then Propagate: Graph Neural Networks meet Personalized
        PageRank" <https://arxiv.org/abs/1810.05997>`
        Tensorflow 1.x implementation: <https://github.com/klicperajo/ppnp>
        Pytorch implementation: <https://github.com/klicperajo/ppnp>
    """

    def data_step(self,
                  adj_transform="PPR",
                  attr_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(attr_transform)(graph.attr_matrix)

        feat, adj = gf.astensors(attr_matrix, adj_matrix, device=self.data_device)

        # ``adj`` and ``feat`` are cached for later use
        self.register_cache(feat=feat, adj=adj)

    def model_step(self,
                   hids=[64],
                   acts=['relu'],
                   ppr_dropout=0.,
                   dropout=0.5,
                   bias=True):

        model = models.APPNP(self.graph.num_feats,
                             self.graph.num_classes,
                             hids=hids,
                             acts=acts,
                             ppr_dropout=ppr_dropout,
                             dropout=dropout,
                             bias=bias,
                             approximated=False)

        return model

    def config_train_data(self, index):

        labels = self.graph.label[index]
        sequence = FullBatchSequence(inputs=[self.cache.feat, self.cache.adj],
                                     y=labels,
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
