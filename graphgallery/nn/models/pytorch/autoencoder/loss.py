import torch

EPS = 1e-7

REDUCTION = {'mean': torch.mean, 'sum': torch.sum, 'none': lambda x: x}

class LogLoss(torch.nn.Module):
    def __init__(self, pos_weight=1.0, reduction='mean'):
        super().__init__()
        assert reduction in REDUCTION
        self.pos_weight = pos_weight
        self.reduction = REDUCTION[reduction]

    def forward(self, output, labels):
        pos_loss = -labels * torch.log(output + EPS)
        neg_loss = - (1 - labels) * torch.log(1 - output + EPS)
        loss = self.pos_weight * pos_loss + neg_loss
        return self.reduction(loss)
