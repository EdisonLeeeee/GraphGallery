import torch
import torch.nn.functional as F


def l2_normalize(d):
    return F.normalize(d, p=2, dim=1)


def get_normalized_vector(d):
    d /= 1e-12 + d.abs().max(dim=1).values.view(-1, 1)
    d /= 1e-6 + d.pow(2.0).sum(dim=1).view(-1, 1)
    return d


def masked_kld_with_logits(logit_q, logit_p, mask=None):
    q = F.softmax(logit_q, dim=-1)
    cross_entropy = softmax_cross_entropy_with_logits(logits=logit_p, labels=q)
    if mask is not None:
        mask = mask / mask.mean()
        cross_entropy *= mask
    return cross_entropy.mean()


def softmax_cross_entropy_with_logits(labels, logits, dim=-1):
    return (-labels * F.log_softmax(logits, dim=dim)).sum(dim=dim)
