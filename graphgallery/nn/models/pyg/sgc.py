import torch.nn as nn
from torch_geometric.nn import SGConv


class SGC(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 hids=[],
                 acts=[],
                 K=2,
                 dropout=0.,
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
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.dropout(x)
        x = self.conv(x, edge_index, edge_weight)
        return x

    def cache_clear(self):
        self.conv._cached_x = None
        return self
