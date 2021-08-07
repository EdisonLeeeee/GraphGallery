import torch

EPS = 1e-7
MAX_LOGSTD = 100

REDUCTION = {'mean': torch.mean, 'sum': torch.sum, 'none': lambda x: x}


class BCELoss(torch.nn.Module):
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

    def extra_repr(self):
        return f"pos_weight={self.pos_weight:.3f}"


class BCELossWithKLLoss(torch.nn.Module):
    def __init__(self, pos_weight=1.0, reduction='mean'):
        super().__init__()
        assert reduction in REDUCTION
        self.pos_weight = pos_weight
        self.reduction = REDUCTION[reduction]

    def forward(self, output, labels):
        if self.training:
            output, mu, logstd = output
            # logstd = logstd.clamp(max=MAX_LOGSTD)
            kl_loss = -0.5 / mu.size(0) * self.reduction(torch.sum(1 + 2 * logstd - mu.pow(2) - logstd.exp().pow(2), dim=1))
        else:
            kl_loss = 0.

        pos_loss = -labels * torch.log(output + EPS)
        neg_loss = -(1 - labels) * torch.log(1 - output + EPS)
        loss = self.pos_weight * pos_loss + neg_loss
        bce_loss = self.reduction(loss)

        return bce_loss + kl_loss

    def extra_repr(self):
        return f"pos_weight={self.pos_weight:.3f}"
