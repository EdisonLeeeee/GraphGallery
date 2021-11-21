import torch
import graphgallery.nn.models.dgl as models
from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import NodeClasTrainer
from graphgallery.gallery.nodeclas import DGL


@DGL.register()
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
        feat, g = gf.astensors(attr_matrix, adj_matrix, device=self.data_device)

        # ``g`` and ``feat`` are cached for later use
        self.register_cache(feat=feat, g=g)

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
        sequence = FullBatchSequence(inputs=[self.cache.feat, self.cache.g],
                                     y=labels,
                                     out_index=index,
                                     device=self.data_device,
                                     escape=type(self.cache.g))
        return sequence

    def config_optimizer(self) -> torch.optim.Optimizer:
        lr = self.cfg.get('lr', 0.01)
        weight_decay = self.cfg.get('weight_decay', 1e-4)
        return torch.optim.Adam(self.model.parameters(),
                                weight_decay=weight_decay, lr=lr)
