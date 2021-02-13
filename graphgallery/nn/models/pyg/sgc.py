import torch.nn as nn
from torch import optim

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.metrics.pytorch import Accuracy

from torch_geometric.nn import SGConv


class SGC(TorchKeras):
    def __init__(self,
                 in_features,
                 out_features,
                 hids=[],
                 acts=[],
                 K=2,
                 dropout=None,
                 weight_decay=5e-5,
                 lr=0.2,
                 bias=False):
        super().__init__()

        if hids or acts:
            raise RuntimeError(
                f"Arguments 'hids' and 'acts' are not supported to use in SGC (PyG backend)."
            )

        # assert dropout, "unused"
        conv = SGConv(in_features,
                      out_features,
                      bias=bias,
                      K=K,
                      cached=True,
                      add_self_loops=True)
        self.conv = conv
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam(conv.parameters(),
                                          lr=lr,
                                          weight_decay=weight_decay),
                     metrics=[Accuracy()])

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv(x, edge_index, edge_weight)
        return x
