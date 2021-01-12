from graphgallery.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery import PyTorch
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model


@PyTorch.register()
class SimPGCN(Trainer):
    """
        Implementation of Similarity Preserving Graph Convolutional Networks (SimPGCN).
        `Node Similarity Preserving Graph Convolutional Networks
        <https://arxiv.org/abs/2011.09643>`
        Pytorch implementation: <https://github.com/ChandlerBang/SimP-GCN>

        Create a SimPGCN model.

        This can be instantiated in the following way:

            trainer = SimPGCN(graph)
                with a `graphgallery.data.Graph` instance representing
                A sparse, attributed, labeled graph.
    """    
    def process_step(self,
                     adj_transform="normalize_adj",
                     attr_transform=None,
                     graph_transform=None):

        graph = gf.get(graph_transform)(self.graph)
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        node_attr = gf.get(attr_transform)(graph.node_attr)

        knn_graph = gf.normalize_adj(gf.knn_graph(node_attr), fill_weight=0.)
        pseudo_labels, node_pairs = gf.attr_sim(node_attr)
        
        X, A, knn_graph, pseudo_labels = gf.astensors(node_attr, adj_matrix, knn_graph, pseudo_labels, device=self.device)
        
        # ``A``, ``X`` and ``knn_graph`` are cached for later use
        self.register_cache(X=X, A=A, knn_graph=knn_graph, 
                           pseudo_labels=pseudo_labels, node_pairs=node_pairs)

    def builder(self,
                hids=[64],
                acts=['relu'],
                dropout=0.5,
                lambda_=5.0,
                gamma=0.1,                
                weight_decay=5e-4,
                lr=0.01,
                use_bias=False):

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
                      use_bias=use_bias)

        return model

    def train_sequence(self, index):
        
        labels = self.graph.node_label[index]
        cache = self.cache
        sequence = FullBatchSequence(x=[cache.X, cache.A, cache.knn_graph, 
                                        cache.pseudo_labels, cache.node_pairs],
                                     y=labels,
                                     out_weight=index,
                                     device=self.device)
        return sequence
    
    def test_sequence(self, index):
        
        labels = self.graph.node_label[index]
        sequence = FullBatchSequence(x=[self.cache.X, self.cache.A],
                                     y=labels,
                                     out_weight=index,
                                     device=self.device)
        return sequence   

