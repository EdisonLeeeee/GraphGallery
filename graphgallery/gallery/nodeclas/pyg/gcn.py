from graphgallery.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyG
from graphgallery.gallery.nodeclas import Trainer
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
        node_attr = gf.get(attr_transform)(graph.node_attr)

        X, E = gf.astensors(node_attr, adj_matrix, device=self.data_device)

        # ``E`` and ``X`` are cached for later use
        self.register_cache(X=X, E=E)

    def model_step(self,
                   hids=[16],
                   acts=['relu'],
                   dropout=0.5,
                   weight_decay=5e-4,
                   lr=0.01,
                   bias=True):

        model = get_model("GCN", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      hids=hids,
                      acts=acts,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      bias=bias)

        return model

    def train_loader(self, index):

        labels = self.graph.node_label[index]
        sequence = FullBatchSequence([self.cache.X, *self.cache.E],
                                     labels,
                                     out_weight=index,
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
        node_attr = gf.get(attr_transform)(graph.node_attr)

        X, E = gf.astensors(node_attr, adj_matrix, device=self.data_device)

        # ``E`` and ``X`` are cached for later use
        self.register_cache(X=X, E=E)

    def model_step(self,
                   hids=[16],
                   acts=['relu'],
                   dropout=0.5,
                   weight_decay=5e-4,
                   lr=0.01,
                   bias=True,
                   p=0.05):

        model = get_model("DropEdge", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      p=p,
                      hids=hids,
                      acts=acts,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      bias=bias)

        return model

    def train_loader(self, index):

        labels = self.graph.node_label[index]
        sequence = FullBatchSequence([self.cache.X, *self.cache.E],
                                     labels,
                                     out_weight=index,
                                     device=self.data_device)
        return sequence
