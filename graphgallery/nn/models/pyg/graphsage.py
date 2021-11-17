import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from graphgallery.nn.layers.pytorch import activations
from graphgallery.nn.metrics import Accuracy
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(self, in_features, out_features,
                 hids=[32], acts=['relu'], dropout=0.5,
                 weight_decay=5e-4, lr=0.01, bias=False,
                 output_normalize=False):

        super().__init__()
        conv = nn.ModuleList()
        act_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        for hid, act in zip(hids, acts):
            conv.append(SAGEConv(in_features,
                                 hid,
                                 bias=bias))
            act_layers.append(activations.get(act))
            in_features = hid
        conv.append(SAGEConv(in_features, out_features, bias=bias))

        self.conv = conv
        self.acts = act_layers
        self.output_normalize = output_normalize
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam(conv.parameters(),
                                          weight_decay=weight_decay, lr=lr),
                     metrics=[Accuracy()])

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        num_layers = len(adjs)
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.conv[i]((x, x_target), edge_index)
            if i != num_layers - 1:
                x = self.acts[i](x)
                x = self.dropout(x)
        if self.output_normalize:
            x = F.normalize(x, dim=1, p=2)
        return x
