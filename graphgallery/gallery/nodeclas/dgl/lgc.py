from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model

from graphgallery.gallery.nodeclas import DGL


@DGL.register()
class LGC(Trainer):
    """
        Implementation of Linear Graph Convolution (LGC). 
        `Simple Graph Convolutional Networks <https://arxiv.org/abs/2106.05809>`
        Pytorch implementation: <https://github.com/lpasa/LGC>

    """

    def data_step(self,
                  adj_transform="add_self_loop",
                  attr_transform=None):
        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(attr_transform)(graph.attr_matrix)
        X, G = gf.astensors(attr_matrix, adj_matrix, device=self.data_device)

        # ``G`` and ``X`` are cached for later use
        self.register_cache(X=X, G=G)

    def model_step(self,
                   hids=[],
                   acts=[],
                   dropout=0.,
                   K=20,
                   weight_decay=5e-5,
                   lr=0.2,
                   bias=True):

        model = get_model("LGC", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      hids=hids,
                      acts=acts,
                      K=K,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      bias=bias)

        return model

    def train_loader(self, index):
        labels = self.graph.label[index]
        sequence = FullBatchSequence(inputs=[self.cache.X, self.cache.G],
                                     y=labels,
                                     out_index=index,
                                     device=self.data_device,
                                     escape=type(self.cache.G))
        return sequence


@DGL.register()
class EGC(LGC):
    """
        Implementation of Exponential Graph Convolution (EGC). 
        `Simple Graph Convolutional Networks <https://arxiv.org/abs/2106.05809>`
        Pytorch implementation: <https://github.com/lpasa/LGC>

    """

    def model_step(self,
                   hids=[],
                   acts=[],
                   dropout=0.,
                   K=5,
                   weight_decay=5e-4,
                   lr=0.2,
                   bias=True):

        model = get_model("EGC", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      hids=hids,
                      acts=acts,
                      K=K,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      bias=bias)

        return model


@DGL.register()
class hLGC(Trainer):
    """
        Implementation of Hyper Linear Graph Convolution (hLGC). 
        `Simple Graph Convolutional Networks <https://arxiv.org/abs/2106.05809>`
        Pytorch implementation: <https://github.com/lpasa/LGC>

    """

    def data_step(self,
                  adj_transform="add_self_loop",
                  attr_transform=None):
        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(attr_transform)(graph.attr_matrix)
        X, G = gf.astensors(attr_matrix, adj_matrix, device=self.data_device)

        # ``G`` and ``X`` are cached for later use
        self.register_cache(X=X, G=G)

    def model_step(self,
                   hids=[],
                   acts=[],
                   dropout=0.,
                   K=10,
                   weight_decay=5e-4,
                   lr=0.01,
                   bias=True):

        model = get_model("hLGC", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      hids=hids,
                      acts=acts,
                      K=K,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      bias=bias)

        return model

    def train_loader(self, index):
        labels = self.graph.label[index]
        sequence = FullBatchSequence(inputs=[self.cache.X, self.cache.G],
                                     y=labels,
                                     out_index=index,
                                     device=self.data_device,
                                     escape=type(self.cache.G))
        return sequence
