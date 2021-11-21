import torch
import dgl.function as fn
from torch import nn
from torch.nn import functional as F


class DAGNNConv(nn.Module):
    def __init__(self,
                 in_features,
                 out_features=1,
                 K=10,
                 bias=False):
        super().__init__()
        assert out_features == 1, "'out_features' must be 1"
        self.in_features = in_features
        self.out_features = out_features
        self.lin = nn.Linear(in_features, out_features, bias=bias)
        self.K = K
        self.act = nn.Sigmoid()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, graph, x):

        with graph.local_scope():
            results = [x]

            degs = graph.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm = norm.to(x.device).unsqueeze(1)

            for _ in range(self.K):
                x = x * norm
                graph.ndata['h'] = x
                graph.update_all(fn.copy_u('h', 'm'),
                                 fn.sum('m', 'h'))
                x = graph.ndata['h']
                x = x * norm
                results.append(x)

            H = torch.stack(results, dim=1)
            S = self.act(self.lin(H))
            S = S.permute(0, 2, 1)
            H = torch.matmul(S, H).squeeze()

            return H

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features}, K={self.K})"
