from graphgallery.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model


@PyTorch.register()
class MLP(Trainer):

    def data_step(self,
                  attr_transform=None):

        graph = self.graph
        node_attr = gf.get(attr_transform)(graph.node_attr)

        X = gf.astensors(node_attr, device=self.data_device)

        # ``A`` and ``X`` are cached for later use
        self.register_cache(X=X)

    def model_step(self,
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

    def train_loader(self, index):

        labels = self.graph.node_label[index]
        sequence = FullBatchSequence(x=[self.cache.X],
                                     y=labels,
                                     out_index=index,
                                     device=self.data_device)
        return sequence
