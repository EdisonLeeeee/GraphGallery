import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelPropagation(nn.Module):
    """label propagation model adapted from https://github.com/CUAI/CorrectAndSmooth
    `"Learning from Labeled and
    Unlabeled Datawith Label Propagation"
    <http://mlg.eng.cam.ac.uk/zoubin/papers/CMU-CALD-02-107.pdf>`_ paper

    """

    def __init__(self, num_layers=50, alpha=0.5, residual=True):
        super().__init__()
        self.num_layers = num_layers
        self.alpha = alpha
        self.residual = residual

    @torch.no_grad()
    def forward(self, y, adj, mask=None):
        if y.dtype == torch.long:
            y = F.one_hot(y.view(-1)).float()

        out = y
        if mask is not None:
            out = torch.zeros_like(y)
            out[mask] = y[mask]

        if self.residual:
            res = (1 - self.alpha) * out
        else:
            res = out.clone()

        for _ in range(self.num_layers):
            out = self.alpha * (adj @ out) + res
            out = torch.clamp(out, 0, 1)
        return out
