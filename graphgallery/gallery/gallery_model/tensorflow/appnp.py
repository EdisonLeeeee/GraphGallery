from graphgallery.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery import TensorFlow
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model


@TensorFlow.register()
class APPNP(Trainer):
    """Implementation of approximated personalized propagation of neural 
        predictions (APPNP).
        `Predict then Propagate: Graph Neural Networks meet Personalized
        PageRank" <https://arxiv.org/abs/1810.05997>`
        Tensorflow 1.x implementation: <https://github.com/klicperajo/ppnp>
        Pytorch implementation: <https://github.com/klicperajo/ppnp>
    """

    def process_step(self,
                     adj_transform="normalize_adj",
                     attr_transform=None,
                     graph_transform=None):

        graph = gf.get(graph_transform)(self.graph)
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        node_attr = gf.get(attr_transform)(graph.node_attr)

        X, A = gf.astensors(node_attr, adj_matrix, device=self.device)

        # ``A`` and ``X`` are cached for later use
        self.register_cache(X=X, A=A)

    def builder(self,
                hids=[64],
                acts=['relu'],
                alpha=0.1,
                K=10,
                ppr_dropout=0.2,
                dropout=0.5,
                weight_decay=5e-4,
                lr=0.01,
                use_bias=True,
                use_tfn=True):

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
                      use_bias=use_bias,
                      approximated=True)
        if use_tfn:
            model.use_tfn()

        return model

    def train_sequence(self, index):

        labels = self.graph.node_label[index]
        sequence = FullBatchSequence(x=[self.cache.X, self.cache.A],
                                     y=labels,
                                     out_weight=index,
                                     device=self.device)
        return sequence


@TensorFlow.register()
class PPNP(Trainer):
    """Implementation of exact personalized propagation of neural 
        predictions (PPNP).
        `Predict then Propagate: Graph Neural Networks meet Personalized
        PageRank" <https://arxiv.org/abs/1810.05997>`
        Tensorflow 1.x implementation: <https://github.com/klicperajo/ppnp>
        Pytorch implementation: <https://github.com/klicperajo/ppnp>
    """

    def process_step(self,
                     adj_transform="PPR",
                     attr_transform=None,
                     graph_transform=None):

        graph = gf.get(graph_transform)(self.graph)
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        node_attr = gf.get(attr_transform)(graph.node_attr)

        X, A = gf.astensors(node_attr, adj_matrix, device=self.device)

        # ``A`` and ``X`` are cached for later use
        self.register_cache(X=X, A=A)

    def builder(self,
                hids=[64],
                acts=['relu'],
                ppr_dropout=0.,
                dropout=0.5,
                weight_decay=5e-4,
                lr=0.01,
                use_bias=True,
                use_tfn=True):

        model = get_model("APPNP", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      hids=hids,
                      acts=acts,
                      ppr_dropout=ppr_dropout,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      use_bias=use_bias,
                      approximated=False)
        if use_tfn:
            model.use_tfn()
        return model

    def train_sequence(self, index):

        labels = self.graph.node_label[index]
        sequence = FullBatchSequence(x=[self.cache.X, self.cache.A],
                                     y=labels,
                                     out_weight=index,
                                     device=self.device)
        return sequence
