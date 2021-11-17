from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyG
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model


@PyG.register()
class GCN(Trainer):
    """
        Implementation of Graph Convolutional Networks (GCN). 
        `Semi-Supervised Classification with Graph Convolutional Networks 
        <https://arxiv.org/abs/1609.02907>`
        Tensorflow 1.x implementation: <https://github.com/tkipf/gcn>
        Pytorch implementation: <https://github.com/tkipf/pygcn>

    """

    def data_step(self,
                  adj_transform="normalize_adj",
                  attr_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(attr_transform)(graph.attr_matrix)

        feat, edges = gf.astensors(attr_matrix, adj_matrix, device=self.data_device)

        # ``edges`` and ``feat`` are cached for later use
        self.register_cache(feat=feat, edges=edges)

    def model_step(self,
                   hids=[16],
                   acts=['relu'],
                   dropout=0.5,
                   weight_decay=5e-4,
                   lr=0.01,
                   bias=True):

        model = get_model("GCN", self.backend)
        model = model(self.graph.num_feats,
                      self.graph.num_classes,
                      hids=hids,
                      acts=acts,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      bias=bias)

        return model

    def train_loader(self, index):

        labels = self.graph.label[index]
        sequence = FullBatchSequence([self.cache.feat, *self.cache.edges],
                                     labels,
                                     out_index=index,
                                     device=self.data_device)
        return sequence


@PyG.register()
class DropEdge(Trainer):
    """
        Implementation of Graph Convolutional Networks (GCN) with DropEdge. 
    """

    def data_step(self,
                  adj_transform="normalize_adj",
                  attr_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(attr_transform)(graph.attr_matrix)

        feat, edges = gf.astensors(attr_matrix, adj_matrix, device=self.data_device)

        # ``edges`` and ``feat`` are cached for later use
        self.register_cache(feat=feat, edges=edges)

    def model_step(self,
                   hids=[16],
                   acts=['relu'],
                   dropout=0.5,
                   weight_decay=5e-4,
                   lr=0.01,
                   bias=True,
                   p=0.3):

        model = get_model("DropEdge", self.backend)
        model = model(self.graph.num_feats,
                      self.graph.num_classes,
                      p=p,
                      hids=hids,
                      acts=acts,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      bias=bias)

        return model

    def train_loader(self, index):

        labels = self.graph.label[index]
        sequence = FullBatchSequence([self.cache.feat, *self.cache.edges],
                                     labels,
                                     out_index=index,
                                     device=self.data_device)
        return sequence


@PyG.register()
class RDrop(Trainer):
    """
        Implementation of Graph Convolutional Networks (GCN) with R-Drop regularization
        in `R-Drop: Regularized Dropout for Neural Networks<https://arxiv.org/abs/2106.14448>`__
        See: https://github.com/dropreg/R-Drop
    """

    def data_step(self,
                  adj_transform="normalize_adj",
                  attr_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(attr_transform)(graph.attr_matrix)

        feat, edges = gf.astensors(attr_matrix, adj_matrix, device=self.data_device)

        # ``edges`` and ``feat`` are cached for later use
        self.register_cache(feat=feat, edges=edges)

    def model_step(self,
                   hids=[16],
                   acts=['relu'],
                   dropout=0.5,
                   weight_decay=5e-4,
                   lr=0.01,
                   kl=0.005,
                   bias=True,
                   p=0.3):

        model = get_model("RDrop", self.backend)
        model = model(self.graph.num_feats,
                      self.graph.num_classes,
                      p=p,
                      kl=kl,
                      hids=hids,
                      acts=acts,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      bias=bias)

        return model

    def train_loader(self, index):

        labels = self.graph.label[index]
        sequence = FullBatchSequence([self.cache.feat, *self.cache.edges],
                                     labels,
                                     out_index=index,
                                     device=self.data_device)
        return sequence
