import torch
from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model

from graphgallery.gallery.nodeclas import DGL


@DGL.register()
class ALaGAT(Trainer):
    """
        Implementation of ALaGAT in
        `When Do GNNs Work: Understanding and Improving Neighborhood Aggregation 
        <https://www.ijcai.org/Proceedings/2020/0181.pdf>`
        DGL implementation: <https://github.com/raspberryice/ala-gcn>

    """

    def data_step(self,
                  adj_transform="add_self_loop",
                  feat_transform=None):
        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(feat_transform)(graph.attr_matrix)
        X, G = gf.astensors(attr_matrix, adj_matrix, device=self.data_device)

        # ``G`` and ``X`` are cached for later use
        self.register_cache(X=X, G=G)

    def model_step(self,
                   hids=[8] * 5,
                   acts=[None] * 5,
                   num_heads=8,
                   dropout=0.5,
                   weight_decay=5e-4,
                   lr=0.005,
                   binary_reg=0.,
                   share_tau=True,
                   bias=False):

        model = get_model("ala_gnn.ALaGAT", self.backend)
        model = model(self.graph.num_feats,
                      self.graph.num_classes,
                      self.graph.num_nodes,
                      hids=hids,
                      acts=acts,
                      num_heads=num_heads,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      binary_reg=binary_reg,
                      share_tau=share_tau,
                      lr=lr,
                      bias=bias)

        return model

    def train_loader(self, index):
        labels = self.graph.label[index]
        self.model.cache.update(x=self.cache.X.to(self.device), y=gf.astensor(labels, device=self.device))
        sequence = FullBatchSequence([self.cache.X, self.cache.G],
                                     labels,
                                     out_index=index,
                                     device=self.data_device,
                                     escape=type(self.cache.G))
        return sequence

    def test_loader(self, index):
        labels = self.graph.label[index]
        sequence = FullBatchSequence([self.cache.X, self.cache.G],
                                     labels,
                                     out_index=index,
                                     device=self.device,
                                     escape=type(self.cache.G))
        return sequence
