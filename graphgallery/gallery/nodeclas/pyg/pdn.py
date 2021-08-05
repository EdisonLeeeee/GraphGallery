from graphgallery.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyG
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model


@PyG.register()
class PDN(Trainer):
    """
        Implementation of Pathfinder Discovery Networks (PDN). 
        `Pathfinder Discovery Networks for Neural Message Passing 
        <https://arxiv.org/abs/2010.12878>`
        Pytorch implementation: <https://github.com/benedekrozemberczki/PDN>

    """

    def data_step(self,
                  edge_transform=None,
                  attr_transform=None,
                  edge_attr_transform=None):

        graph = self.graph
        edge_index, edge_weight = gf.get(edge_transform)(graph.edge_index, graph.edge_weight)
        node_attr = gf.get(attr_transform)(graph.node_attr)
        edge_attr = gf.get(edge_attr_transform)(graph.edge_attr)

        X, edge_index, edge_x = gf.astensors(node_attr,
                                             edge_index,
                                             edge_attr,
                                             device=self.data_device)
        self.register_cache(X=X, edge_index=edge_index, edge_x=edge_x)

    def model_step(self,
                   hids=[32],
                   acts=['relu'],
                   pdn_hids=32,
                   dropout=0.5,
                   weight_decay=5e-5,
                   lr=0.01,
                   bias=True):

        model = get_model("PDN", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      self.graph.num_edge_attrs,
                      hids=hids,
                      pdn_hids=pdn_hids,
                      acts=acts,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      bias=bias)

        return model

    def train_loader(self, index):

        labels = self.graph.node_label[index]
        sequence = FullBatchSequence([self.cache.X,
                                      self.cache.edge_index,
                                      self.cache.edge_x],
                                     labels,
                                     out_index=index,
                                     device=self.data_device)
        return sequence
