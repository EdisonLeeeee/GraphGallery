import torch
import graphgallery.nn.models.pytorch as models
from graphgallery.data.sequence import FastGCNBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from graphgallery.gallery.nodeclas import NodeClasTrainer


@PyTorch.register()
class FastGCN(NodeClasTrainer):
    """
        Implementation of Fast Graph Convolutional Networks (FastGCN).
        `FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling 
        <https://arxiv.org/abs/1801.10247>`
        Tensorflow 1.x implementation: <https://github.com/matenure/FastGCN>

    """

    def data_step(self,
                  adj_transform="normalize_adj",
                  feat_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(feat_transform)(graph.attr_matrix)
        attr_matrix = adj_matrix @ attr_matrix

        feat, adj = gf.astensor(attr_matrix, device=self.data_device), adj_matrix

        # ``adj`` and ``feat`` are cached for later use
        self.register_cache(feat=feat, adj=adj)

    def model_step(self,
                   hids=[32],
                   acts=['relu'],
                   dropout=0.5,
                   bias=False):

        model = models.FastGCN(self.graph.num_feats,
                               self.graph.num_classes,
                               hids=hids,
                               acts=acts,
                               dropout=dropout,
                               bias=bias)
        return model

    def config_train_data(self, index):
        batch_size = self.cfg.get('batch_size_train', 256)
        num_samples = self.cfg.get('num_samples_train', 50)

        labels = self.graph.label[index]
        adj_matrix = self.graph.adj_matrix[index][:, index]
        adj_matrix = self.transform.adj_transform(adj_matrix)

        feat = self.cache.feat[index]
        sequence = FastGCNBatchSequence(inputs=[feat, adj_matrix],
                                        nodes=index,
                                        y=labels,
                                        batch_size=batch_size,
                                        num_samples=num_samples,
                                        device=self.data_device)
        return sequence

    def config_test_data(self, index):
        batch_size = self.cfg.get('batch_size_test', None)
        num_samples = self.cfg.get('num_samples_test', None)
        labels = self.graph.label[index]
        adj = self.cache.adj[index]

        sequence = FastGCNBatchSequence(inputs=[self.cache.feat, adj],
                                        nodes=index,
                                        y=labels,
                                        batch_size=batch_size,
                                        num_samples=num_samples,
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
