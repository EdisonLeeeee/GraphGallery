
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.utils import softmax
    from torch_scatter import scatter
    pyg_enabled = True
except ImportError:
    pyg_enabled = False


def cosine_similarity(A, B):
    inner_product = (A * B).sum(1)
    C = inner_product / (torch.norm(A, 2, 1) * torch.norm(B, 2, 1) + 1e-7)
    return C


class AGNNConv(nn.Module):

    def __init__(self, trainable=True):
        super().__init__()

        if trainable:
            # unifrom initialization
            self.beta = nn.Parameter(torch.tensor(1.).uniform_(0, 1))
        else:
            self.beta = torch.tensor(1.)

        self.trainable = trainable

    def forward_pyg(self, x, adj):
        row, col = adj.coalesce().indices()
        A = x[row]
        B = x[col]

        sim = self.beta * cosine_similarity(A, B)
        P = softmax(sim, row)
        src = x[row] * P.view(-1, 1)
        out = scatter(src, col, dim=0, reduce="add")
        return out

    def forward_py(self, x, adj):
        if adj.is_sparse:
            adj = adj.to_dense()
        # add a minor constant (1e-7) to denominator to prevent division by
        # zero error
        if self.trainable:
            norm2 = torch.norm(x, 2, 1).view(-1, 1)
            cos = self.beta * torch.div(torch.mm(x, x.t()), torch.mm(norm2, norm2.t()) + 1e-7)
        else:
            cos = 0.

        # neighborhood masking (inspired by this repo:
        # https://github.com/danielegrattarola/keras-gat)
        mask = (1. - adj) * -1e9
        masked = cos + mask

        # propagation matrix
        P = F.softmax(masked, dim=1)

        # attention-guided propagation
        output = torch.mm(P, x)
        return output

    if pyg_enabled:
        forward = forward_pyg
    else:
        forward = forward_py

    def extra_repr(self):
        return f"trainable={self.trainable}"
