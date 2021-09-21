from graphgallery.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model


@PyTorch.register()
class SimPGCN(Trainer):
    """
        Implementation of Similarity Preserving Graph Convolutional Networks (SimPGCN).
        `Node Similarity Preserving Graph Convolutional Networks
        <https://arxiv.org/abs/2011.09643>`
        Pytorch implementation: <https://github.com/ChandlerBang/SimP-GCN>
    """

    def data_step(self,
                  adj_transform="normalize_adj",
                  attr_transform=None,
                  recalculate=True):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        node_attr = gf.get(attr_transform)(graph.node_attr)

        X, A = gf.astensors(node_attr, adj_matrix, device=self.data_device)

        # ``A`` and ``X`` are cached for later use
        self.register_cache(X=X, A=A)

        if recalculate:
            # Uses this to save time for structure evation attack
            # NOTE: Please make sure the node attribute matrix remains the same if recalculate=False
            knn_graph = gf.normalize_adj(gf.knn_graph(node_attr), fill_weight=0.)
            pseudo_labels, node_pairs = gf.attr_sim(node_attr)
            knn_graph, pseudo_labels = gf.astensors(knn_graph, pseudo_labels, device=self.data_device)

            self.register_cache(knn_graph=knn_graph, pseudo_labels=pseudo_labels, node_pairs=node_pairs)

    def model_step(self,
                   hids=[64],
                   acts=[None],
                   dropout=0.5,
                   lambda_=5.0,
                   gamma=0.1,
                   weight_decay=5e-4,
                   lr=0.01,
                   bias=False):

        model = get_model("SimPGCN", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      hids=hids,
                      acts=acts,
                      lambda_=lambda_,
                      gamma=gamma,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      bias=bias)

        return model

    def train_loader(self, index):

        labels = self.graph.node_label[index]
        cache = self.cache
        sequence = FullBatchSequence(inputs=[cache.X, cache.A, cache.knn_graph,
                                             cache.pseudo_labels, cache.node_pairs],
                                     y=labels,
                                     out_index=index,
                                     device=self.data_device)
        return sequence

    def test_loader(self, index):

        labels = self.graph.node_label[index]
        sequence = FullBatchSequence(inputs=[self.cache.X, self.cache.A],
                                     y=labels,
                                     out_index=index,
                                     device=self.data_device)
        return sequence
