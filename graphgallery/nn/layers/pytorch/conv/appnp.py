import torch.nn as nn
from ..dropout import MixedDropout


class PPNProp(nn.Module):
    def __init__(self, dropout: float = 0.):
        super().__init__()
        if not dropout:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(dropout)

    def forward(self, x, adj):
        return self.dropout(adj).mm(x)

    def __repr__(self):
        return f"{self.__class__.__name__}(dropout={self.dropout})"


class APPNProp(nn.Module):
    def __init__(self, alpha: float = 0.1, K: int = 10, dropout: float = 0.):
        super().__init__()
        self.alpha = alpha
        self.K = K
        if not dropout:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(dropout)

    def forward(self, x, adj):
        h = x
        for _ in range(self.K):
            A_drop = self.dropout(adj)
            h = (1 - self.alpha) * A_drop.mm(h) + self.alpha * x
        return h

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha}, K={self.K}, dropout={self.dropout})"
