import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        x_coal = x.coalesce()
        drop_val = F.dropout(x_coal._values(), self.p, self.training)
        return torch.sparse.FloatTensor(x_coal._indices(), drop_val, x.shape)


class MixedDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.dense_dropout = nn.Dropout(p)
        self.sparse_dropout = SparseDropout(p)

    def forward(self, x):
        if x.is_sparse:
            return self.sparse_dropout(x)
        else:
            return self.dense_dropout(x)
