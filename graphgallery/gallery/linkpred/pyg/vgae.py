import numpy as np
from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.linkpred import PyG
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model


@PyG.register()
class VGAE(Trainer):
    """Implementation of Variational Graph AutoEncoder (VGAE) in
    `Variational Graph Auto-Encoders
    <https://arxiv.org/abs/1611.07308>`
    TensorFlow 1.x implementation <https://github.com/tkipf/gae>
    """

    def data_step(self,
                  attr_transform=None):

        graph = self.graph
        # adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(attr_transform)(graph.attr_matrix)

        X = gf.astensor(attr_matrix, device=self.data_device)

        # ``X`` is cached for later use
        self.register_cache(X=X)

    def model_step(self,
                   out_features=16,
                   hids=[32],
                   acts=['relu'],
                   dropout=0.,
                   weight_decay=0.,
                   lr=0.01,
                   bias=False):

        model = get_model("VGAE", self.backend)
        model = model(self.graph.num_feats,
                      out_features=out_features,
                      hids=hids,
                      acts=acts,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      bias=bias)

        return model

    def train_loader(self, edge_index):
        if isinstance(edge_index, (list, tuple)):
            train_edges = edge_index[0]  # postive edge index
        else:
            train_edges = edge_index

        train_edges = gf.astensor(train_edges, device=self.data_device)

        self.register_cache(E=train_edges)
        sequence = FullBatchSequence([self.cache.X, train_edges],
                                     out_index=edge_index,
                                     device=self.data_device)
        return sequence

    def test_loader(self, edge_index):

        if isinstance(edge_index, (list, tuple)):
            edge_index = np.hstack(edge_index)

        y = self.graph.adj_matrix[edge_index[0], edge_index[1]].A1
        y = np.clip(y, 0, 1)

        sequence = FullBatchSequence([self.cache.X, self.cache.E],
                                     y=y,
                                     out_index=edge_index,
                                     device=self.data_device)
        return sequence

    def predict_loader(self, edge_index):
        if isinstance(edge_index, (list, tuple)):
            edge_index = np.hstack(edge_index)

        sequence = FullBatchSequence([self.cache.X, self.cache.E],
                                     out_index=edge_index,
                                     device=self.data_device)
        return sequence
