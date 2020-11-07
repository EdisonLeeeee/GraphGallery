from graphgallery.nn.gallery import SemiSupervisedModel
from graphgallery.nn.models.dgl_torch import GCN as dglGCN
from graphgallery.sequence import FullBatchNodeSequence
from graphgallery import functional as F

class GCN(SemiSupervisedModel):
    def __init__(self, *graph, adj_transform="normalize_adj", attr_transform="normalize_attr",
                 device='cpu:0', seed=None, name=None, **kwargs):
        super().__init__(*graph, device=device, seed=seed, name=name, **kwargs)

        self.adj_transform = F.get(adj_transform)
        self.attr_transform = F.get(attr_transform)
        self.process()

    def process_step(self):
        graph = self.graph
        adj_matrix = self.adj_transform(graph.adj_matrix)
        attr_matrix = self.attr_transform(graph.attr_matrix)

        self.feature_inputs, self.structure_inputs = F.astensors(attr_matrix, adj_matrix, device=self.device)

    @F.EqualVarLength()
    def build(self, hiddens=[16], activations=['relu'], dropout=0.5,
              weight_decay=5e-4, lr=0.01, use_bias=False):

        self.model = dglGCN(self.graph.n_attrs, self.graph.n_classes,
                            hiddens=hiddens, activations=activations, dropout=dropout,
                            weight_decay=weight_decay, lr=lr, use_bias=use_bias
                            ).to(self.device)

    def train_sequence(self, index):

        labels = self.graph.labels[index]
        sequence = FullBatchNodeSequence(
            [self.feature_inputs, self.structure_inputs, index], labels,
            device=self.device, escape=type(self.structure_inputs))
        return sequence
