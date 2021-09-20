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

    def train_loader(self, index, block_size=2000):

        labels = self.graph.node_label[index]
        sequence = DenseBatchSequence(inputs=[self.cache.X, self.cache.A],
                                      y=labels,
                                      out_index=index,
                                      block_size=block_size,
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

    def __init__(self, inputs, y=None, out_index=None, block_size=2000, device='cpu', escape=None, **kwargs):
        dataset = gf.astensors(inputs, y, out_index, device=device, escape=escape)
        super().__init__([dataset], batch_size=None, collate_fn=self.collate_fn, device=device, escape=escape, **kwargs)
        self.block_size = block_size

    def collate_fn(self, dataset):
        (x, adj), y, out_index = dataset
        rand_indx = self.astensor(np.random.choice(np.arange(adj.shape[0]), self.block_size))
        if out_index is not None:
            rand_indx[:out_index.size(0)] = out_index
        features_batch = x[rand_indx]
        adj_label_batch = adj[rand_indx, :][:, rand_indx]
        return (features_batch, adj_label_batch), y, out_index
