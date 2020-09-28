import torch
import torch.nn.functional as F

from torch.nn import Module, ModuleList
from torch import optim

from graphgallery.nn.models import SemiSupervisedModel
from graphgallery.nn.models import TorchKerasModel
from graphgallery.nn.models.get_activation import get_activation
from graphgallery.nn.layers import GraphConvolution
from graphgallery.sequence import FullBatchNodeSequence
from graphgallery.utils.decorators import EqualVarLength

from graphgallery import transformers as T


class _Model(TorchKerasModel):

    def __init__(self, in_channels, hiddens,
                 out_channels, n_heads=[8], activations=['elu'],
                 dropouts=[0.6], l2_norms=[5e-4],
                 lr=0.01, use_bias=True):

        super().__init__()

        # save for later usage
        self.dropouts = dropouts

        self.gcs = ModuleList()
        self.acts = []
        paras = []

        inc = in_channels
        for hidden, n_head, act, l2_norm in zip(hiddens, n_heads, activations, l2_norms):
            # 缺少dropout
            layer = GraphAttention(inc, hidden, attn_heads=n_head, attn_heads_reduction='concat', activation=act, use_bias=use_bias)
            self.gcs.append(layer)
            self.acts.append(get_activation(act))
            paras.append(dict(params=layer.parameters(), weight_decay=l2_norm))
            inc = hidden

        layer = GraphAttention(inc, out_channels, attn_heads=1, attn_heads_reduction='average', use_bias=use_bias)
        self.gcs.append(layer)
        # do not use weight_decay in the final layer
        paras.append(dict(params=layer.parameters(), weight_decay=0.))

        self.optimizer = optim.Adam(paras, lr=lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, inputs):
        x, adj, idx = inputs

        for i in range(len(self.gcs) - 1):
            act = self.acts[i]
            x = act(self.gcs[i]([x, adj]))
            x = F.dropout(x, self.dropouts[i], training=self.training)

        x = self.gcs[-1]([x, adj])  # last layer

        if idx is None:
            return x
        else:
            return x[idx]

    def reset_parameters(self):
        for i, l in enumerate(self.gcs):
            self.gcs[i].reset_parameters()

class GAT(SemiSupervisedModel):
    """
    """

    def __init__(self, *graph, adj_transformer="add_selfloops", attr_transformer=None,
                 device='cpu:0', seed=None, name=None, **kwargs):
        """
        """

        super().__init__(*graph, device=device, seed=seed, name=name, **kwargs)

        self.adj_transformer = T.get(adj_transformer)
        self.attr_transformer = T.get(attr_transformer)
        self.process()

    def process_step(self):
        graph = self.graph
        adj_matrix = self.adj_transformer(graph.adj_matrix)
        attr_matrix = self.attr_transformer(graph.attr_matrix)

        self.feature_inputs, self.structure_inputs = T.astensors(
            attr_matrix, adj_matrix)

    # use decorator to make sure all list arguments have the same length
    @EqualVarLength(include=["n_heads"])
    def build(self, hiddens=[8], n_heads=[8], activations=['elu'], dropouts=[0.6],
              l2_norms=[5e-4], lr=0.01, use_bias=True):

        self.model = _Model(self.graph.n_attrs, hiddens, self.graph.n_classes,
                            n_heads=n_heads, activations=activations, dropouts=dropouts, l2_norms=l2_norms,
                            lr=lr, use_bias=use_bias).to(self.device)

    def train_sequence(self, index):
        index = T.asintarr(index)
        labels = self.graph.labels[index]
        sequence = FullBatchNodeSequence(
            [self.feature_inputs, self.structure_inputs, index], labels, device=self.device)

        return sequence
