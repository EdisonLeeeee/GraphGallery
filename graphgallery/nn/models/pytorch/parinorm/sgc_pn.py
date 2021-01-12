import torch.nn as nn
from torch import optim
from graphgallery.nn.models import TorchKeras
from graphgallery.nn.layers.pytorch.get_activation import get_activation
from graphgallery.nn.layers.pytorch import PairNorm
from graphgallery.nn.metrics.pytorch import Accuracy


class SGC_PN(TorchKeras):
    """PairNorm: Tackling Oversmoothing in GNNs
    <https://openreview.net/forum?id=rkecl1rtwB>
    ICLR 2020"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 hids=[],
                 acts=[],
                 order=2,
                 norm_mode=None,
                 norm_scale=10,
                 dropout=0.6,
                 weight_decay=5e-4,
                 lr=0.005,
                 use_bias=False):

        super().__init__()
        assert not hids and not acts
        self.linear = nn.Linear(in_channels, out_channels, bias=use_bias)
        self.norm = PairNorm(norm_mode, norm_scale)
        self.dropout = nn.Dropout(p=dropout)
        self.order = order
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam(self.parameters(), lr=0.01),
                     metrics=[Accuracy()])

    def forward(self, x, adj):
        x = self.norm(x)
        for _ in range(self.order):
            x = adj.mm(x)
            x = self.norm(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x
