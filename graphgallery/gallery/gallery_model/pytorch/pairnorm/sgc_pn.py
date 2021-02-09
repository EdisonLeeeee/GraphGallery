from graphgallery.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery import PyTorch
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model


@PyTorch.register()
class SGC_PN(Trainer):

    def process_step(self,
                     adj_transform=("normalize_adj", dict(symmetric=False)),
                     attr_transform=None,
                     graph_transform=None):

        graph = gf.get(graph_transform)(self.graph)
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        node_attr = gf.get(attr_transform)(graph.node_attr)

        X, A = gf.astensors(node_attr, adj_matrix, device=self.device)

        # ``A`` and ``X`` are cached for later use
        self.register_cache(X=X, A=A)

    def builder(self,
                hids=[],
                acts=[],
                K=2,
                norm_mode="PN",
                norm_scale=10,
                dropout=0.6,
                weight_decay=5e-4,
                lr=0.005,
                bias=True):

        model = get_model("SGC_PN", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      hids=hids,
                      acts=acts,
                      K=K,
                      norm_mode=norm_mode,
                      norm_scale=norm_scale,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      bias=bias)

        return model

    def train_sequence(self, index):

        labels = self.graph.node_label[index]
        sequence = FullBatchSequence(x=[self.cache.X, self.cache.A],
                                     y=labels,
                                     out_weight=index,
                                     device=self.device)
        return sequence
