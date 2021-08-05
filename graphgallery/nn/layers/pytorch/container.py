import inspect
import torch.nn as nn


class Sequential(nn.Sequential):
    def __init__(self, *args, reverse=False):
        super().__init__(*args)
        self.reverse = reverse

    def forward(self, *input):
        input, others = split_input(input, reverse=self.reverse)
        for module in self:
            assert hasattr(module, 'forward')
            para_required = len(inspect.signature(module.forward).parameters)
            if para_required == 1 and not isinstance(module, (MIMO, Tree)):
                input = module(input)
            else:
                if self.reverse:
                    input = module(*others, input)
                else:
                    input = module(input, *others)

            if isinstance(input, tuple) and not isinstance(module, (MIMO, Tree)):
                input, others = split_input(input, reverse=self.reverse)

        return input


class MIMO(nn.Sequential):
    """Multiple inputs and multiple outputs from multiple mudule"""

    def forward(self, *input):
        for module in self:
            input = module(*input)
            if not isinstance(input, tuple):
                input = (input,)

        if len(input) == 1:
            input, = input
        return input


class Tree(nn.Sequential):
    """Single input and multiple outputs from multiple mudule

    >>> t = Tree(nn.Linear(10, 3), nn.Linear10, 5))
    >>> x = torch.randn(10, 10)
    >>> t(x) # outputs (out1, out2)

    """

    def forward(self, *input):
        out = tuple(module(*input) for module in self)
        if len(out) == 1:
            out, = out
        return out

    def __repr__(self):
        out = f"{self.__class__.__name__}("
        middle = ""
        for i, module in enumerate(self):
            middle += f"\n(input) --> {module} --> (output{i})"
        if middle:
            middle += "\n"
        out += middle + ")"
        return out


def split_input(input, reverse=True):
    if reverse:
        *others, input = input
    else:
        input, *others = input
    return input, others
