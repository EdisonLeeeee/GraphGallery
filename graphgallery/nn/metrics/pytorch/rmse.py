import torch
from torch import nn
from .metric import Metric


class RMSE(Metric):

    def __init__(self, name="rmse", reduction="mean", **kwargs):
        super().__init__(name=name, **kwargs)
        assert reduction in {"sum", "mean"}
        self.reduction = reduction

    def forward(self, y_true, y_pred):
        return self.update_state(y_true, y_pred)

    @torch.no_grad()
    def update_state(self, y_true, y_pred):
        assert y_pred.size() == y_true.size(), f'Size {y_pred.size()} is not equal to {y_true.size()}'

        self._targets.append(y_true)
        self._predictions.append(y_pred)

    def reset_states(self):
        self._targets = []
        self._predictions = []

    def result(self):
        _predictions = torch.cat(self._predictions, dim=0)
        _targets = torch.cat(self._targets, dim=0)
        result = (_predictions - _targets).square()

        if self.reduction == 'mean':
            result = result.mean()
        elif self.reduction == 'sum':
            result = result.sum()
        else:
            # won't be never heppen
            pass

        return result.sqrt()
