import torch
import graphgallery.nn.models.pytorch as models
from graphgallery.nn.layers.pytorch import SSGConv
from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from graphgallery.gallery import Trainer


@PyTorch.register()
class SSGC(Trainer):
    """
        Implementation of Simple Spectral Graph Convolution (SSGC). 
        `Simple Spectral Graph Convolution <https://openreview.net/forum?id=CYO5T-YjWZV>`
        Pytorch implementation: https://github.com/allenhaozhu/SSGC      

    """

    def data_step(self,
                  adj_transform="normalize_adj",
                  attr_transform=None,
                  K=16,
                  alpha=0.1):
        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(attr_transform)(graph.attr_matrix)

        feat, adj = gf.astensors(attr_matrix, adj_matrix, device=self.data_device)

        feat = SSGConv(K=K, alpha=alpha)(feat, adj)
        # ``adj`` and ``feat`` are cached for later use
        self.register_cache(feat=feat, adj=adj)

    def model_step(self,
                   hids=[],
                   acts=[],
                   dropout=0.5,
                   bias=True):

        model = models.MLP(self.graph.num_feats,
                           self.graph.num_classes,
                           hids=hids,
                           acts=acts,
                           dropout=dropout,
                           bias=bias)

        return model

    def config_train_data(self, index):
        labels = self.graph.label[index]
        feat = self.cache.feat[index]
        sequence = FullBatchSequence(feat, labels, device=self.data_device)
        return sequence

    def config_optimizer(self) -> torch.optim.Optimizer:
        lr = self.cfg.get('lr', 0.2)
        weight_decay = self.cfg.get('weight_decay', 5e-5)
        model = self.model
        return torch.optim.Adam(model.parameters(),
                                weight_decay=weight_decay, lr=lr)
