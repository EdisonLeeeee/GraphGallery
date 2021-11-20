import torch
import graphgallery.nn.models.dgl as models
from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import NodeClasTrainer
from graphgallery.gallery.nodeclas import DGL


@DGL.register()
class JKNet(NodeClasTrainer):
    """
        Implementation of Jumping Knowledge Networks (JKNet). 
        `Representation Learning on Graphs with Jumping Knowledge Networks
        <https://arxiv.org/abs/1806.03536>`

        DGL implementation: <https://github.com/mori97/JKNet-dgl>
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
                   hids=[16] * 5,
                   acts=['relu'] * 5,
                   mode='cat',
                   dropout=0.5,
                   bias=True):

        model = models.JKNet(self.graph.num_feats,
                             self.graph.num_classes,
                             hids=hids,
                             acts=acts,
                             mode=mode,
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
        lr = self.cfg.get('lr', 0.005)
        weight_decay = self.cfg.get('weight_decay', 5e-4)
        return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
