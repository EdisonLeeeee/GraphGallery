from torch.nn import Module


class SGConvolution(Module):
    def __init__(self, K=2, **kwargs):
        super().__init__()
        self.K = K

    def forward(self, x, adj):

        for _ in range(self.K):
            x = adj.mm(x)

        return x

    def reset_parameters(self):
        pass

    def extra_repr(self):
        return f"K={self.K}"
