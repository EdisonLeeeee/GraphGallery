import torch
from torch import nn
from .metric import Metric


class Accuracy(Metric):
    """Calculates how often predictions equal labels.

      This metric creates two local variables, `total` and `count` that are used to
      compute the frequency with which `y_pred` matches `y_true`. This frequency is
      ultimately returned as `binary accuracy`: an idempotent operation that simply
      divides `total` by `count`.

      If `sample_weight` is `None`, weights default to 1.
      Use `sample_weight` of 0 to mask values.

      Standalone usage:

      >>> m = gg.nn.metrics.Accuracy()
      >>> m.update_state(torch.tensor([1, 2, 3, 4]), torch.tensor([0, 2, 3, 4]))
      >>> m.result()
      0.75

      >>> m.reset_state()
      >>> m.update_state(torch.tensor([1, 2, 3, 4]), torch.tensor([0, 2, 3, 4]),
      ...                sample_weight=torch.tensor([1, 1, 0, 0]))
      >>> m.result()
      0.5
    """

    def __init__(self, name="accuracy", reduction="mean", **kwargs):
        super().__init__(name=name, **kwargs)
        assert reduction in {"sum", "mean"}
        self.reduction = reduction

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
            sample_weight = sample_weight.bool()
            y_pred = y_pred[sample_weight]
            y_true = y_true[sample_weight]

        self.correct += torch.sum(y_pred == y_true)
        self.total += y_true.numel()

    def reset_states(self):
        self.total = torch.tensor(0)
        self.correct = torch.tensor(0)

    def result(self):
        if self.total == 0: return None
        if self.reduction == 'mean':
            return self.correct.float() / self.total
        elif self.reduction == 'sum':
            return self.correct
        else:
            # won't be never heppen
            pass
