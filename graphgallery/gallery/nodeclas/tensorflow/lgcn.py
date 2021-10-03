import numpy as np

from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import TensorFlow
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model


@TensorFlow.register()
class LGCN(Trainer):
    """
        Implementation of Large-Scale Learnable Graph Convolutional Networks (LGCN).
        `Large-Scale Learnable Graph Convolutional Networks <https://arxiv.org/abs/1808.03965>`
        Tensorflow 1.x implementation: <https://github.com/divelab/lgcn>
    """

    def data_step(self,
                  adj_transform="normalize_adj",
                  attr_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix).toarray()
        node_attr = gf.get(attr_transform)(graph.node_attr)

        # ``A`` and ``X`` are cached for later use
        self.register_cache(X=node_attr, A=adj_matrix)

    def model_step(self,
                   hids=[32],
                   num_filters=[8, 8],
                   acts=[None, None],
                   dropout=0.8,
                   weight_decay=5e-4,
                   lr=0.1,
                   bias=False,
                   K=8, exclude=["num_filters", "acts"]):

        model = get_model("LGCN", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      hids=hids,
                      acts=acts,
                      dropout=dropout,
                      num_filters=num_filters,
                      weight_decay=weight_decay,
                      lr=lr,
                      bias=bias,
                      K=K)

        return model

    def train_loader(self, index, batch_size=np.inf):
        cache = self.cache
        mask = gf.index_to_mask(index, self.graph.num_nodes)
        index = get_indice_graph(cache.A, index, batch_size)
        while index.size < self.cfg.model.K:
            index = get_indice_graph(cache.A, index)

        A = cache.A[index][:, index]
        X = cache.X[index]
        mask = mask[index]
        labels = self.graph.node_label[index[mask]]

        sequence = FullBatchSequence([X, A],
                                     labels,
                                     out_index=mask,
                                     device=self.data_device)
        return sequence


def get_indice_graph(adj_matrix, indices, size=np.inf, dropout=0.):
    if dropout > 0.:
        indices = np.random.choice(indices, int(indices.size * (1 - dropout)),
                                   False)
    neighbors = adj_matrix[indices].sum(axis=0).nonzero()[0]
    if neighbors.size > size - indices.size:
        neighbors = np.random.choice(list(neighbors), size - len(indices),
                                     False)
    indices = np.union1d(indices, neighbors)
    return indices
