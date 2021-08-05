import torch
from torch import nn
from .metric import Metric


class F1Score(Metric):

    def __init__(self, name="F1", from_logits=False, average='micro', **kwargs):
        super().__init__(name=name, **kwargs)
        self.from_logits = from_logits
        self.average = average

    def forward(self, y_true, y_pred, sample_weight=None):
        return self.update_state(y_true, y_pred, sample_weight=sample_weight)

    @torch.no_grad()
    def update_state(self, y_true, y_pred, sample_weight=None):

        if y_pred.ndim == 2:
            y_pred = y_pred.argmax(1)
        if y_true.ndim == 2:
            y_true = y_true.argmax(1)

        assert y_pred.size() == y_true.size(), f'Size {y_pred.size()} is not equal to {y_true.size()}'

        if sample_weight is not None:
            assert y_pred.size() == sample_weight.size(), f'Size {y_pred.size()} is not equal to {sample_weight.size()}'
            # TODO
            raise NotImplementedError

        if self.from_logits:
            y_pred = torch.sigmoid(y_pred)

        self._targets.append(y_true)
        self._predictions.append(y_pred)

    def reset_states(self):
        self._targets = []
        self._predictions = []

    def result(self):
        _predictions = torch.cat(self._predictions, dim=0)
        _targets = torch.cat(self._targets, dim=0)

        from sklearn.metrics import f1_score

        y_true = _targets.cpu().numpy()
        y_pred = _predictions.cpu().numpy()
        return f1_score(y_true, y_pred, average=self.average)
