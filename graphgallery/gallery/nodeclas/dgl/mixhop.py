import torch
import graphgallery.nn.models.dgl as models
from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import NodeClasTrainer
from graphgallery.gallery.nodeclas import DGL


@DGL.register()
class MixHop(NodeClasTrainer):
    """
        Implementation of MixHop Graph Convolutional layer from paper `MixHop: Higher-Order Graph Convolutional Architecturesvia Sparsified Neighborhood Mixing
        <https://arxiv.org/abs/1905.00067>`
        Tensorflow  implementation: <https://github.com/samihaija/mixhop>
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

    def model_step(self, hids=[60] * 3,
                   acts=['tanh'] * 3,
                   p=[0, 1, 2],
                   dropout=0.5,
                   bias=False):

        model = models.MixHop(self.graph.num_feats,
                              self.graph.num_classes,
                              hids=hids,
                              acts=acts,
                              dropout=dropout,
                              p=p,
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
        lr = self.cfg.get('lr', 0.1)
        weight_decay = self.cfg.get('weight_decay', 5e-4)
        return torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def config_scheduler(self, optimizer: torch.optim.Optimizer):
        step_size = self.cfg.get('step_size', 40)
        gamma = self.cfg.get('gamma', 0.01)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=gamma)
