import torch
import torch.nn as nn


class LabelPropagation(nn.Module):
    """label propagation model from https://github.com/CUAI/CorrectAndSmooth"""

    def __init__(self, num_propagations=50, alpha=0.5, residual=True):
        super().__init__()
        self.num_propagations = num_propagations
        self.alpha = alpha
        self.residual = residual

    def forward(self, y, adj, idx_know):
        result = torch.zeros_like(y).float()
        result[idx_know] = y[idx_know]
        y = result.clone()

        alpha = self.alpha
        residual = self.residual
        for _ in range(self.num_propagations):
            result = alpha * (adj @ result)
            if residual:
                result += (1 - alpha) * y
            else:
                result += y
            result = torch.clamp(result, 0, 1)
        return result
