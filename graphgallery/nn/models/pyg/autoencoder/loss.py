
import torch

EPS = 1e-7


class LogLoss(torch.nn.Module):
    def __init__(self, pos_weight=1.0, reduction='mean'):
        super().__init__()
        assert reduction in {'mean', 'sum', None}
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, pos_pred, neg_pred):
        pos_loss = -torch.log(pos_pred + EPS)
        neg_loss = -torch.log(1 - neg_pred + EPS)
        loss = self.pos_weight * pos_loss + neg_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
