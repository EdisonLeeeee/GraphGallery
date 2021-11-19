import torch.nn as nn
from graphgallery.nn.layers.pytorch import PairNorm


class SGC_PN(nn.Module):
    """PairNorm: Tackling Oversmoothing in GNNs
    <https://openreview.net/forum?id=rkecl1rtwB>
    ICLR 2020"""

    def __init__(self,
                 in_features,
                 out_features,
                 hids=[],
                 acts=[],
                 K=2,
                 norm_mode=None,
                 norm_scale=10,
                 dropout=0.6,
                 bias=False):

        super().__init__()
        assert not hids and not acts
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.norm = PairNorm(norm_mode, norm_scale)
        self.dropout = nn.Dropout(p=dropout)
        self.K = K

    def forward(self, x, adj):
        x = self.norm(x)
        for _ in range(self.K):
            x = adj.mm(x)
            x = self.norm(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x
