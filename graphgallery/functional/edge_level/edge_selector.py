import numpy as np
from .to_adj import asedge
__all__ = ["threshold_selector", "percent_selector"]


def threshold_selector(edge, score, threshold=0., return_score=False):
    edge = asedge(edge, shape="row_wise")  # shape [M, 2]
    assert edge.shape[0] == score.shape[0]
    ix = np.where(score <= threshold)[0]
    if not return_score:
        return edge[ix]
    else:
        return edge[ix], score[ix]


def percent_selector(edge, score, percent=0.01, threshold=None, return_score=False):
    edge = asedge(edge, shape="row_wise")  # shape [M, 2]
    assert edge.shape[0] == score.shape[0]
    n = int(score.shape[0] * percent)
    if threshold is not None:
        n = min(n, (score <= threshold).sum())
    idx = np.argsort(score)[:n]

    if not return_score:
        return edge[idx]
    else:
        return edge[idx], score[idx]
