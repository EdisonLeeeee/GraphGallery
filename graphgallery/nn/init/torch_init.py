import math

import torch

from graphgallery.typing import TorchTensor

__all__ = ['uniform', 'kaiming_uniform',
           'xavier_uniform',
           'glorot_uniform',
           'glorot_orthogonal', 'zeros',
           'ones', 'normal', 'reset']

xavier_uniform = torch.nn.init.xavier_uniform_
xavier_normal = torch.nn.init.xavier_normal_
kaiming_uniform = torch.nn.init.kaiming_uniform_
kaiming_normal = torch.nn.init.kaiming_normal_
uniform = torch.nn.init.uniform_
normal = torch.nn.init.normal_


def uniform(tensor) -> TorchTensor:
    if tensor is not None:
        bound = 1.0 / math.sqrt(tensor.size(-1))
        tensor.data.uniform_(-bound, bound)


# def kaiming_uniform(tensor, fan, a):
#     if tensor is not None:
#         bound = math.sqrt(6 / ((1 + a**2) * fan))
#         tensor.data.uniform_(-bound, bound)


def glorot_uniform(tensor, scale=6.0) -> TorchTensor:
    if tensor is not None:
        stdv = math.sqrt(scale / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def glorot_orthogonal(tensor, scale) -> TorchTensor:
    if tensor is not None:
        torch.nn.init.orthogonal_(tensor.data)
        scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
        tensor.data *= scale.sqrt()


def zeros(tensor) -> TorchTensor:
    if tensor is not None:
        tensor.data.fill_(0)


def ones(tensor) -> TorchTensor:
    if tensor is not None:
        tensor.data.fill_(1)


def normal(tensor, mean, std) -> TorchTensor:
    if tensor is not None:
        tensor.data.normal_(mean, std)


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)
