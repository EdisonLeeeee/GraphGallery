import torch
import torch.nn.functional as F


def get_normalized_vector(d):
    d /= 1e-12 + d.abs().max(dim=1).values.view(-1, 1)
    d /= 1e-6 + d.pow(2.0).sum(dim=1).view(-1, 1)
    return d


def kld_with_logits(logit_q, logit_p):
    q = F.softmax(logit_q, dim=-1)
    cross_entropy = softmax_cross_entropy_with_logits(logits=logit_p, labels=q)
    entropy = softmax_cross_entropy_with_logits(logits=logit_q, labels=q)
    return (cross_entropy - entropy).mean()


def neighbor_kld_with_logit(neighbor_logits, p_logit):
    dist = 0.
    for neighbor_logit in neighbor_logits:
        dist += kld_with_logits(neighbor_logit, p_logit)
    return dist


def softmax_cross_entropy_with_logits(labels, logits, dim=-1):
    return (-labels * F.log_softmax(logits, dim=dim)).sum(dim=dim)


def l2_normalize(d):
    norm = torch.norm(d, p=2, dim=1, keepdim=True)
    norm = d / norm
    norm[torch.isnan(norm)] = 0.
    return norm
