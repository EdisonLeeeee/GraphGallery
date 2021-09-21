import torch
from graphgallery.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model

from graphgallery.gallery.nodeclas import DGL


@DGL.register()
class ALaGCN(Trainer):
    """
        Implementation of ALaGCN in
        `When Do GNNs Work: Understanding and Improving Neighborhood Aggregation 
        <https://www.ijcai.org/Proceedings/2020/0181.pdf>`
        DGL implementation: <https://github.com/raspberryice/ala-gcn>

    """

    def data_step(self,
                  adj_transform="add_selfloops",
                  attr_transform=None):
        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        node_attr = gf.get(attr_transform)(graph.node_attr)
        X, G = gf.astensors(node_attr, adj_matrix, device=self.data_device)

        # ``G`` and ``X`` are cached for later use
        self.register_cache(X=X, G=G)

    def model_step(self,
                   hids=[16] * 5,
                   acts=[None] * 5,
                   dropout=0.5,
                   weight_decay=5e-6,
                   lr=0.01,
                   binary_reg=0.,
                   share_tau=True,
                   bias=False):

        model = get_model("ala_gnn.ALaGCN", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      self.graph.num_nodes,
                      hids=hids,
                      acts=acts,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      binary_reg=binary_reg,
                      share_tau=share_tau,
                      lr=lr,
                      bias=bias)

        return model

    def train_loader(self, index):
        labels = self.graph.node_label[index]
        self.model.cache.update(x=self.cache.X.to(self.device), y=gf.astensor(labels, device=self.device))
        sequence = FullBatchSequence([self.cache.X, self.cache.G],
                                     labels,
                                     out_index=index,
                                     device=self.data_device,
                                     escape=type(self.cache.G))
        return sequence

    def test_loader(self, index):
        labels = self.graph.node_label[index]
        sequence = FullBatchSequence([self.cache.X, self.cache.G],
                                     labels,
                                     out_index=index,
                                     device=self.device,
                                     escape=type(self.cache.G))
        return sequence
