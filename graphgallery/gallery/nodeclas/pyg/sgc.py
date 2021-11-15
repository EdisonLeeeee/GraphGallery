from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyG
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model


@PyG.register()
class SGC(Trainer):
    """
        Implementation of Simplifying Graph Convolutional Networks (SGC).
        `Simplifying Graph Convolutional Networks <https://arxiv.org/abs/1902.07153>`
        Pytorch implementation: <https://github.com/Tiiiger/SGC>

    """

    def data_step(self,
                  adj_transform=None,
                  attr_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(attr_transform)(graph.attr_matrix)

        X, E = gf.astensors(attr_matrix, adj_matrix, device=self.data_device)

        # ``E`` and ``X`` are cached for later use
        self.register_cache(X=X, E=E)

    def model_step(self,
                   hids=[],
                   acts=[],
                   dropout=0.,
                   weight_decay=5e-5,
                   lr=0.2,
                   bias=True,
                   K=2):

        model = get_model("SGC", self.backend)
        model = model(self.graph.num_feats,
                      self.graph.num_classes,
                      K=K,
                      acts=acts,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      bias=bias)

        return model

    def train_loader(self, index):

        labels = self.graph.label[index]
        sequence = FullBatchSequence([self.cache.X, *self.cache.E],
                                     labels,
                                     out_index=index,
                                     device=self.data_device)
        return sequence
