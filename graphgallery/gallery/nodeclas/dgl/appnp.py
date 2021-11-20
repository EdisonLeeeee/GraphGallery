import graphgallery.nn.models.dgl as models
from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import NodeClasTrainer

from graphgallery.gallery.nodeclas import DGL


@DGL.register()
class APPNP(NodeClasTrainer):
    """Implementation of approximated personalized propagation of neural 
        predictions (APPNP).
        `Predict then Propagate: Graph Neural Networks meet Personalized
        PageRank" <https://arxiv.org/abs/1810.05997>`
        Tensorflow 1.x implementation: <https://github.com/klicperajo/ppnp>
        Pytorch implementation: <https://github.com/klicperajo/ppnp>
    """

    def data_step(self,
                  adj_transform="normalize_adj",
                  feat_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(feat_transform)(graph.attr_matrix)
        feat, g = gf.astensors(attr_matrix, adj_matrix, device=self.data_device)

        # ``g`` and ``feat`` are cached for later use
        self.register_cache(feat=feat, g=g)

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
