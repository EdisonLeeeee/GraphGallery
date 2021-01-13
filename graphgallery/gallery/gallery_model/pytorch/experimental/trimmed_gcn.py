from graphgallery.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery import PyTorch
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model

@PyTorch.register()
class TrimmedGCN(Trainer):
    
    def process_step(self,
                     adj_transform="normalize_adj",
                     attr_transform=None,
                     graph_transform=None):

        graph = gf.get(graph_transform)(self.graph)
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        node_attr = gf.get(attr_transform)(graph.node_attr)

        X, A = gf.astensors(node_attr, adj_matrix.tolil().rows, device=self.device)

        # ``A`` and ``X`` are cached for later use
        self.register_cache(X=X, A=A)
        
    def builder(self,
                hids=[64],
                acts=['relu'],
                dropout=0.5,
                weight_decay=5e-5,
                lr=0.01,
                tperc=0.3,
                use_bias=False):
        
        model = get_model("TrimmedGCN", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      hids=hids,
                      acts=acts,
                      tperc=tperc,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      use_bias=use_bias)
        return model
    
    def train_sequence(self, index):

        labels = self.graph.node_label[index]
        sequence = FullBatchSequence([self.cache.X, self.cache.A],
                                     labels,
                                     out_weight=index,
                                     device=self.device)
        return sequence
