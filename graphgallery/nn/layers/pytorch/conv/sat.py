import torch
import torch.nn as nn


class SpectralEigenConv(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=False,
                 K=10,
                 alpha=0.1,
                 **kwargs):
        super().__init__()
        assert K > 0
        self.K = K
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features
        self.w = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, U, V=None):
        """
        x: node attribute matrix
        if `V=None`:
            U: (N, N) adjacency matrix
        else:
            U: (N, k) eigenvector matrix
            V: (k,) eigenvalue
        """
        x = self.w(x)
        if V is not None:
            V_pow = torch.ones_like(V)
            V_out = torch.zeros_like(V)
            for _ in range(self.K):
                V_pow *= V
                V_out += (1 - self.alpha) * V_pow
            V_out = V_out / self.K
            x_out = (U * V_out) @ (U.t() @ x) + self.alpha * x
        else:
            adj = U
            x_in = x
            x_out = torch.zeros_like(x)
            for _ in range(self.K):
                x = torch.spmm(adj, x)
                x_out += (1 - self.alpha) * x
            x_out /= self.K
            x_out += self.alpha * x_in
        return x_out

    def reset_parameters(self):
        self.w.reset_parameters()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features}, K={self.K}, alpha={self.alpha})"


class EigenConv(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=False,
                 K=2,
                 **kwargs):
        super().__init__()
        assert K > 0
        self.K = K
        self.in_features = in_features
        self.out_features = out_features
        self.w = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, U, V=None):
        """
        x: node attribute matrix
        if `V=None`:
            U: (N, N) adjacency matrix
        else:
            U: (N, k) eigenvector matrix
            V: (k,) eigenvalue
        """
        out = self.w(x)
        if V is not None:
            V = V.pow(self.K)
            out = (U * V) @ (U.t() @ out)
        else:
            adj = U
            for _ in range(self.K):
                out = adj.mm(out)
        return out

    def reset_parameters(self):
        self.w.reset_parameters()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features}, K={self.K})"
    
class GraphEigenConv(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=False,
                 **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, U, V=None):
        """
        x: node attribute matrix
        if `V=None`:
            U: (N, N) adjacency matrix
        else:
            U: (N, k) eigenvector matrix
            V: (k,) eigenvalue
        """
        out = self.w(x)
        if V is not None:
            out = (U * V) @ (U.t() @ out)
        else:
            adj = U
            out = adj.mm(out)
        return out

    def reset_parameters(self):
        self.w.reset_parameters()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"
    
