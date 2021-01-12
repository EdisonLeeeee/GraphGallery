from graphgallery.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery import PyG
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model


@PyG.register()
class SGC(Trainer):
    """
        Implementation of Simplifying Graph Convolutional Networks (SGC).
        `Simplifying Graph Convolutional Networks <https://arxiv.org/abs/1902.07153>`
        Pytorch implementation: <https://github.com/Tiiiger/SGC>

    """

    def process_step(self,
                     adj_transform=None,
                     attr_transform=None,
                     graph_transform=None):

        graph = gf.get(graph_transform)(self.graph)
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        node_attr = gf.get(attr_transform)(graph.node_attr)

        X, E = gf.astensors(node_attr, adj_matrix, device=self.device)

        # ``E`` and ``X`` are cached for later use
        self.register_cache(X=X, E=E)

    def builder(self,
                hids=[],
                acts=[],
                dropout=0.,
                weight_decay=5e-5,
                lr=0.2,
                use_bias=True,
                order=2):

        model = get_model("SGC", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      K=order,
                      acts=acts,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      use_bias=use_bias)

        return model

    def train_sequence(self, index):

        labels = self.graph.node_label[index]
        sequence = FullBatchSequence([self.cache.X, *self.cache.E],
                                     labels,
                                     out_weight=index,
                                     device=self.device)
        return sequence
