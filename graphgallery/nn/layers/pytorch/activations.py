import torch
import torch.nn as nn
import torch.nn.functional as F

act_dict = dict(relu="ReLU",
                relu6="ReLU6",
                sigmoid="Sigmoid",
                celu="CELU",
                elu="ELU",
                gelu="GELU",
                leakyrelu="LeakyReLU",
                prelu="PReLU",
                selu="SELU",
                silu="SiLU",
                softmax="Softmax",
                tanh="Tanh")


class Noop(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()

    def forward(self, x):
        return x


def get(act, inplace=False):
    if act is None:
        return Noop()

    if isinstance(act, nn.Module):
        return act

    out = act_dict.get(act, None)
    if out:
        return getattr(nn, out)()
    else:
        raise ValueError(f"Unknown activation {act}")


def get_fn(act):
    if callable(act):
        return act

    if act is None:
        return lambda x: x

    if not isinstance(act, str):
        raise ValueError("'act' is expected a 'string', "
                         f"but got {type(act)}.")

    if hasattr(torch, act):
        fn = getattr(torch, act)
    else:
        fn = getattr(F, act, None)

    if fn:
        return fn
    else:
        raise ValueError(f"Unknown activation {act}")
