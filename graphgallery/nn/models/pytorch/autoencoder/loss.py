
import torch

EPS = 1e-7


class LogLoss(torch.nn.Module):
    def __init__(self, pos_weight=1.0, reduction='mean'):
        super().__init__()
        assert reduction in {'mean', 'sum', None}
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, output, labels):
        labels = labels.float()
        pos_loss = -labels * torch.log(output + EPS)
        neg_loss = - (1 - labels) * torch.log(1 - output + EPS)
        loss = self.pos_weight * pos_loss + neg_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
