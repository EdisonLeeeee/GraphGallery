from graphgallery.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from graphgallery.gallery.nodeclas import Trainer
from graphgallery.nn.models import get_model


@PyTorch.register()
class APPNP(Trainer):
    """Implementation of approximated personalized propagation of neural 
        predictions (APPNP).
        `Predict then Propagate: Graph Neural Networks meet Personalized
        PageRank" <https://arxiv.org/abs/1810.05997>`
        Tensorflow 1.x implementation: <https://github.com/klicperajo/ppnp>
        Pytorch implementation: <https://github.com/klicperajo/ppnp>
    """

    def data_step(self,
                  adj_transform="normalize_adj",
                  attr_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        node_attr = gf.get(attr_transform)(graph.node_attr)

        X, A = gf.astensors(node_attr, adj_matrix, device=self.data_device)

        # ``A`` and ``X`` are cached for later use
        self.register_cache(X=X, A=A)

    def model_step(self,
                   hids=[64],
                   acts=['relu'],
                   alpha=0.1,
                   K=10,
                   ppr_dropout=0.,
                   dropout=0.5,
                   weight_decay=5e-4,
                   lr=0.01,
                   bias=True):

        model = get_model("APPNP", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      hids=hids,
                      acts=acts,
                      alpha=alpha,
                      K=K,
                      ppr_dropout=ppr_dropout,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      bias=bias,
                      approximated=True)

        return model

    def train_loader(self, index):

        labels = self.graph.node_label[index]
        sequence = FullBatchSequence(x=[self.cache.X, self.cache.A],
                                     y=labels,
                                     out_weight=index,
                                     device=self.data_device)
        return sequence


@PyTorch.register()
class PPNP(Trainer):
    """Implementation of exact personalized propagation of neural 
        predictions (PPNP).
        `Predict then Propagate: Graph Neural Networks meet Personalized
        PageRank" <https://arxiv.org/abs/1810.05997>`
        Tensorflow 1.x implementation: <https://github.com/klicperajo/ppnp>
        Pytorch implementation: <https://github.com/klicperajo/ppnp>
    """

    def data_step(self,
                  adj_transform="PPR",
                  attr_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        node_attr = gf.get(attr_transform)(graph.node_attr)

        X, A = gf.astensors(node_attr, adj_matrix, device=self.data_device)

        # ``A`` and ``X`` are cached for later use
        self.register_cache(X=X, A=A)

    def model_step(self,
                   hids=[64],
                   acts=['relu'],
                   ppr_dropout=0.,
                   dropout=0.5,
                   weight_decay=5e-4,
                   lr=0.01,
                   bias=True):

        model = get_model("APPNP", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      hids=hids,
                      acts=acts,
                      ppr_dropout=ppr_dropout,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      bias=bias,
                      approximated=False)

        return model

    def train_loader(self, index):

        labels = self.graph.node_label[index]
        sequence = FullBatchSequence(x=[self.cache.X, self.cache.A],
                                     y=labels,
                                     out_weight=index,
                                     device=self.data_device)
        return sequence
