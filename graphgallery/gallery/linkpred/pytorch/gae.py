import numpy as np
from graphgallery.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.linkpred import PyTorch
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model


@PyTorch.register()
class GAE(Trainer):
    """Implementation of Graph AutoEncoder (GAE) in
    `Variational Graph Auto-Encoders
    <https://arxiv.org/abs/1611.07308>`
    TensorFlow 1.x implementation <https://github.com/tkipf/gae>
    """

    def data_step(self,
                  adj_transform="normalize_adj",  # it is required
                  attr_transform=None):

        graph = self.graph
        node_attr = gf.get(attr_transform)(graph.node_attr)

        X = gf.astensor(node_attr, device=self.data_device)

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

        num_edges = self.graph.adj_matrix.sum()
        num_nodes = self.graph.adj_matrix.shape[0]
        pos_weight = (num_nodes**2 - num_edges) / num_edges

        model = get_model("autoencoder.GAE", self.backend)
        model = model(self.graph.num_node_attrs,
                      out_features=out_features,
                      hids=hids,
                      acts=acts,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      pos_weight=pos_weight,
                      bias=bias)

        return model

    def train_loader(self, edge_index):
        if isinstance(edge_index, (list, tuple)):
            train_edges = edge_index[0]  # postive edge index
        else:
            train_edges = edge_index

        full_adj = self.graph.adj_matrix
        edge_weight = full_adj[train_edges[0], train_edges[1]].A1
        adj_matrix = gf.edge_to_sparse_adj(train_edges, edge_weight)
        train_adj = self.transform.adj_transform(adj_matrix)
        adj_label = gf.add_selfloops(adj_matrix).A  # to dense matrix
        train_adj, adj_label = gf.astensors(train_adj, adj_label, device=self.data_device)

        self.register_cache(A=train_adj)
        sequence = FullBatchSequence([self.cache.X, train_adj],
                                     y=adj_label,
                                     device=self.data_device)
        return sequence

    def test_loader(self, edge_index):

        if isinstance(edge_index, (list, tuple)):
            edge_index = np.hstack(edge_index)

        y = self.graph.adj_matrix[edge_index[0], edge_index[1]].A1
        y = np.clip(y, 0, 1)

        sequence = FullBatchSequence([self.cache.X, self.cache.A],
                                     y=y,
                                     out_index=edge_index,
                                     device=self.data_device)
        return sequence

    def predict_loader(self, edge_index):
        if isinstance(edge_index, (list, tuple)):
            edge_index = np.hstack(edge_index)

        sequence = FullBatchSequence([self.cache.X, self.cache.A],
                                     out_index=edge_index,
                                     device=self.data_device)
        return sequence
