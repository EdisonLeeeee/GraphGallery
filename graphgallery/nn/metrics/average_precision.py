import torch
from .metric import Metric


class AveragePrecision(Metric):
    """Computes Average Precision accumulating predictions and the ground-truth during an epoch
    and applying `sklearn.metrics.average_precision_score <https://scikit-learn.org/stable/modules/generated/
    sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score>`_ 
    """

    def __init__(self, name="ap", from_logits=False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.from_logits = from_logits

    def forward(self, y_true, y_pred, sample_weight=None):
        return self.update_state(y_true, y_pred, sample_weight=sample_weight)

    @torch.no_grad()
    def update_state(self, y_true, y_pred, sample_weight=None):
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
        if not self._targets or not self._predictions:
            return None
        _predictions = torch.cat(self._predictions, dim=0)
        _targets = torch.cat(self._targets, dim=0)

        from sklearn.metrics import average_precision_score

        y_true = _targets.cpu().numpy()
        y_pred = _predictions.cpu().numpy()
        return average_precision_score(y_true, y_pred)
