import torch
import graphgallery.nn.models.pytorch as models
from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from graphgallery.gallery.nodeclas import NodeClasTrainer


@PyTorch.register()
class SGC_PN(NodeClasTrainer):

    def data_step(self,
                  adj_transform=("normalize_adj", dict(symmetric=False)),
                  feat_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(feat_transform)(graph.attr_matrix)

        feat, adj = gf.astensors(attr_matrix, adj_matrix, device=self.data_device)

        # ``adj`` and ``feat`` are cached for later use
        self.register_cache(feat=feat, adj=adj)

    def model_step(self,
                   hids=[],
                   acts=[],
                   K=2,
                   norm_mode="PN",
                   norm_scale=10,
                   dropout=0.6,
                   bias=True):

        model = models.SGC_PN(self.graph.num_feats,
                              self.graph.num_classes,
                              hids=hids,
                              acts=acts,
                              K=K,
                              norm_mode=norm_mode,
                              norm_scale=norm_scale,
                              dropout=dropout,
                              bias=bias)

        return model

    def config_train_data(self, index):

        labels = self.graph.label[index]
        sequence = FullBatchSequence(inputs=[self.cache.feat, self.cache.adj],
                                     y=labels,
                                     out_index=index,
                                     device=self.data_device)
        return sequence

    def config_optimizer(self) -> torch.optim.Optimizer:
        lr = self.cfg.get('lr', 0.005)
        weight_decay = self.cfg.get('weight_decay', 5e-4)
        return torch.optim.Adam(self.model.parameters(),
                                weight_decay=weight_decay, lr=lr)
