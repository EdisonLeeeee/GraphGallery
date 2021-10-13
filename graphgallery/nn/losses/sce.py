import torch
import torch.nn as nn
import torch.nn.functional as F


class SCELoss(nn.Module):
    """Symmetric Cross Entropy Loss in
    `Symmetric Cross Entropy for Robust Learning with Noisy Labels<https://arxiv.org/abs/1908.06112>`__
    """

    def __init__(self, a=1, b=1):
        super(SCELoss, self).__init__()
        self.a = a
        self.b = b
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CE
        ce = self.cross_entropy(pred, labels)
        num_classes = pred.size(-1)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-6, max=1.0)
        label_one_hot = F.one_hot(labels, num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))

        loss = self.a * ce + self.b * rce.mean()
        return loss
