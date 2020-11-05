from graphgallery.functional import parse_device
from graphgallery import functional as F
from graphgallery.nn.gallery import SemiSupervisedModel
from graphgallery.nn.models.dgl_pytorch import GCN as dglGCN
from graphgallery.sequence import FullBatchNodeSequence

from dgl import from_scipy

class GCN(SemiSupervisedModel):
    def __init__(self, *graph, adj_transform="normalize_adj", attr_transform="normalize_attr",
                 device='cpu:0', seed=None, name=None, **kwargs):
        super().__init__(*graph, device=device, seed=seed, name=name, **kwargs)

        self.adj_transform = F.get(adj_transform)
        self.attr_transform = F.get(attr_transform)
        self.process()

    def process_step(self):
        graph = self.graph
        attr_matrix = self.attr_transform(graph.attr_matrix)
        adj_matrix = self.adj_transform(graph.adj_matrix)

        self.structure_inputs = from_scipy(adj_matrix).int().to(
            parse_device(self.device)
        )

        self.feature_inputs = F.astensors(attr_matrix, device=self.device)

    @F.EqualVarLength()
    def build(self, hiddens=[16], activations=['relu'], dropout=0.5,
              l2_norm=5e-4, lr=0.01, use_bias=False):
        self.model = dglGCN(self.structure_inputs, self.graph.n_attrs, self.graph.n_classes,
                            hiddens=hiddens, activations=activations, dropout=dropout,
                            l2_norm=l2_norm, lr=lr, use_bias=use_bias
                            ).to(self.device)

    def train_sequence(self, index):

        labels = self.graph.labels[index]
        sequence = FullBatchNodeSequence(
            [self.feature_inputs, index], labels, device=self.device)
        return sequence
