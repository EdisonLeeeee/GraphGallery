from graphgallery.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery import PyTorch
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model


@PyTorch.register()
class MLP(Trainer):

    def process_step(self,
                     attr_transform=None,
                     graph_transform=None):

        graph = gf.get(graph_transform)(self.graph)
        node_attr = gf.get(attr_transform)(graph.node_attr)

        X = gf.astensors(node_attr, device=self.device)

        # ``A`` and ``X`` are cached for later use
        self.register_cache(X=X)

    def builder(self,
                hids=[16],
                acts=['relu'],
                dropout=0.5,
                weight_decay=5e-4,
                lr=0.01,
                bias=False):

        model = get_model("MLP", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      hids=hids,
                      acts=acts,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      bias=bias)

        return model

    def train_sequence(self, index):

        labels = self.graph.node_label[index]
        sequence = FullBatchSequence(x=[self.cache.X],
                                     y=labels,
                                     out_weight=index,
                                     device=self.device)
        return sequence

