from graphgallery.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery import PyG
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
    def process_step(self,
                     adj_transform=None,
                     attr_transform=None,
                     graph_transform=None):

        graph = gf.get(graph_transform)(self.graph)
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        node_attr = gf.get(attr_transform)(graph.node_attr)
        
        edge_index, edge_weight = gf.sparse_adj_to_edge(adj_matrix)
        edge_x = getattr(graph, "edge_attr", edge_weight[..., None])
        edge_index = getattr(graph, "edge_index", edge_index)
        
        X, edge_index, edge_x = gf.astensors(node_attr, 
                                             edge_index, 
                                             edge_x, 
                                             device=self.device)
        self.register_cache(X=X, edge_index=edge_index, edge_x=edge_x)

    def builder(self,
                hids=[32],
                acts=['relu'],
                pdn_hids=32,
                dropout=0.5,
                weight_decay=5e-4,
                lr=0.01,
                bias=True):

        num_edge_attr = self.cache.edge_x.shape[-1]
        model = get_model("PDN", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      num_edge_attr,
                      hids=hids,
                      pdn_hids=pdn_hids,
                      acts=acts,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      bias=bias)

        return model

    def train_sequence(self, index):

        labels = self.graph.node_label[index]
        sequence = FullBatchSequence([self.cache.X, 
                                      self.cache.edge_index, 
                                      self.cache.edge_x],
                                     labels,
                                     out_weight=index,
                                     device=self.device)
        return sequence
