import torch
import numpy as np
from graphgallery.sequence import FullBatchSequence, Sequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model


@PyTorch.register()
class GraphMLP(Trainer):
    """
        Implementation of Graph-MLP.
        `Graph-MLP: Node Classification without Message Passing in Graph
        <https://arxiv.org/abs/2106.04051>`
        Pytorch implementation: <https://github.com/yanghu819/Graph-MLP>

    """

    def data_step(self,
                  adj_transform=("normalize_adj", ("adj_power", dict(power=2)), "to_dense"),
                  attr_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        node_attr = gf.get(attr_transform)(graph.node_attr)

        X, A = gf.astensors(node_attr, adj_matrix, device=self.data_device)

        # ``A`` and ``X`` are cached for later use
        self.register_cache(X=X, A=A)

    def model_step(self,
                   hids=[256],
                   acts=['gelu'],
                   dropout=0.6,
                   weight_decay=5e-3,
                   lr=0.001,
                   bias=True,
                   alpha=10.0,
                   tau=2.0):

        model = get_model("GraphMLP", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      tau=tau,
                      alpha=alpha,
                      hids=hids,
                      acts=acts,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      bias=bias)

        return model

    def train_loader(self, index, batch_size=2000):

        labels = self.graph.node_label[index]
        sequence = DenseBatchSequence(inputs=[self.cache.X, self.cache.A],
                                      y=labels,
                                      out_index=index,
                                      batch_size=batch_size,
                                      device=self.data_device)
        return sequence

    def test_loader(self, index):

        labels = self.graph.node_label[index]
        sequence = FullBatchSequence(inputs=self.cache.X,
                                     y=labels,
                                     out_index=index,
                                     device=self.data_device)
        return sequence


class DenseBatchSequence(Sequence):

    def __init__(self, x, y=None, out_index=None, batch_size=256, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.x = self.astensors(x, device=self.device)
        self.y = self.astensor(y, device=self.device)
        self.out_index = self.astensor(out_index, device=self.device)
        self.batch_size = batch_size

    def __len__(self):
        return 1

    def __getitem__(self, index):
        x, adj = self.x
        rand_indx = self.astensor(np.random.choice(np.arange(adj.shape[0]), self.batch_size), device=self.device)
        if self.out_index is not None:
            rand_indx[:len(self.out_index)] = self.out_index
        features_batch = x[rand_indx]
        adj_label_batch = adj[rand_indx, :][:, rand_indx]
        return (features_batch, adj_label_batch), self.y, self.out_index
