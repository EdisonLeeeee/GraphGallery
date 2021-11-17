import torch.nn as nn
from dgl.nn.pytorch.conv import SGConv


class SGC(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 hids=[],
                 acts=[],
                 K=2,
                 dropout=0.,
                 bias=True):
        super().__init__()

        if hids or acts:
            raise RuntimeError(
                f"Arguments 'hids' and 'acts' are not supported to use in SGC (DGL backend)."
            )

        conv = SGConv(in_features,
                      out_features,
                      bias=bias,
                      k=K,
                      cached=True)
        self.dropout = nn.Dropout(dropout)
        self.conv = conv

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, g):
        x = self.dropout(x)
        x = self.conv(g, x)
        return x

    def cache_clear(self):
        self.conv._cached_h = None
        return self
