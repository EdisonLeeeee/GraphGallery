
import torch

EPS = 1e-7

REDUCTION = {'mean': torch.mean, 'sum': torch.sum, 'none': lambda x: x}


class BCELoss(torch.nn.Module):
    def __init__(self, pos_weight=1.0, reduction='mean'):
        super().__init__()
        assert reduction in REDUCTION
        self.pos_weight = pos_weight
        self.reduction = REDUCTION[reduction]

    def forward(self, pos_pred, neg_pred):
        pos_loss = self.reduction(-torch.log(pos_pred + EPS))
        neg_loss = self.reduction(-torch.log(1 - neg_pred + EPS))
        loss = self.pos_weight * pos_loss + neg_loss
        return loss
