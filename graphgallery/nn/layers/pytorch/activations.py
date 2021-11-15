import torch
import torch.nn as nn
from typing import Optional, Any

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
    def __init__(self, inplace: bool = False):
        super().__init__()

    def forward(self, x: Any) -> Any:
        return x


def get(act: Optional[str] = None, inplace: bool = False) -> torch.nn.Module:
    """get activation functions by `string`

    Example
    -------
    >>> from graphwar.nn import activations
    >>> activations.get('relu')
    ReLU()

    Parameters
    ----------
    act : string or None
        the string to get activations, if `None`, return `Noop` that
        returns the input as output.
    inplace : bool, optional
        the inplace argument in activation functions
        currently it is not work since not all the functions 
        take this argument, by default False

    Returns
    -------
    torch.nn.Module
        the activation function

    Raises
    ------
    ValueError
        unknown or invalid activation string.
    """
    if act is None:
        return Noop()

    if isinstance(act, nn.Module):
        return act

    out = act_dict.get(act, None)
    if out:
        return getattr(nn, out)()
    else:
        raise ValueError(f"Unknown activation {act}. The allowed activation functions are {tuple(act_dict.keys())}.")
