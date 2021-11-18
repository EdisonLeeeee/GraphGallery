import graphgallery.nn.models.pytorch as models
from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from graphgallery.gallery import Trainer


@PyTorch.register()
class TAGCN(Trainer):
    """
        Implementation of Topology Adaptive Graph Convolutional Networks <https://arxiv.org/abs/1710.10370> 
        Tensorflow 1.x implementation: <https://github.com/krohak/TAGCN>
        Create a Topology Adaptive Graph Convolutional Networks
         (TAGCN) model.
    """

    def data_step(self,
                  adj_transform=("normalize_adj",
                                 dict(fill_weight=0.0)),
                  feat_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(feat_transform)(graph.attr_matrix)

        feat, adj = gf.astensors(attr_matrix, adj_matrix, device=self.data_device)

        # ``adj`` and ``feat`` are cached for later use
        self.register_cache(feat=feat, adj=adj)

    def model_step(self,
                   hids=[16],
                   K=3,
                   acts=['relu'],
                   dropout=0.5,
                   bias=True):

        model = models.TAGCN(self.graph.num_feats,
                             self.graph.num_classes,
                             hids=hids,
                             K=K,
                             acts=acts,
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
