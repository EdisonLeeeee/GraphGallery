import graphgallery.nn.models.pytorch as models
from graphgallery.nn.layers.pytorch import SGConv
from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from graphgallery.gallery.nodeclas import NodeClasTrainer


@PyTorch.register()
class SGC(NodeClasTrainer):
    """
        Implementation of Simplifying Graph Convolutional Networks (SGC). 
        `Simplifying Graph Convolutional Networks <https://arxiv.org/abs/1902.07153>`
        Pytorch implementation: <https://github.com/Tiiiger/SGC>

    """

    def data_step(self,
                  adj_transform="normalize_adj",
                  feat_transform=None,
                  K=2):
        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(feat_transform)(graph.attr_matrix)

        feat, adj = gf.astensors(attr_matrix, adj_matrix, device=self.data_device)

        feat = SGConv(K=K)(feat, adj)
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
