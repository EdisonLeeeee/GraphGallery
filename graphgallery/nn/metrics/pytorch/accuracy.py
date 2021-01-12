import torch
from torch import nn
from .metric import Metric


class Accuracy(Metric):

    def __init__(self, name="accuracy", dtype=None,
                 reduction="sum", **kwargs):
        super().__init__(name, dtype, **kwargs)
        assert reduction in {"sum", "mean", "max", "min"}
        # TODO: more reduction
        self.reduction = reduction

    def forward(self, y_true, y_pred,
                out_weight=None):
        return self.update_state(y_true, y_pred,
                                 out_weight=out_weight)

    @torch.no_grad()
    def update_state(self, y_true, y_pred,
                     out_weight=None):
        if out_weight is not None:
            # TODO
            raise NotImplementedError("out_weight")
        if y_pred.ndim == 2:
            y_pred = y_pred.argmax(1)
        self.correct += torch.sum(y_pred == y_true)
        self.total += y_true.numel()

    def reset_states(self):
        # K.batch_set_value([(v, 0) for v in self.variables])
        self.total = torch.tensor(0)
        self.correct = torch.tensor(0)

    def result(self):
        return (self.correct.float() / self.total).detach()
