import inspect
import torch.nn as nn


class Sequential(nn.Sequential):

    def __init__(self, *args, loc=0):
        super().__init__(*args)
        self.loc = loc
        para_required = []
        for module in self:
            assert hasattr(module, "forward"), module
            para_required.append(len(inspect.signature(module.forward).parameters))
        self._para_required = para_required

    def forward(self, *inputs):
        loc = self.loc
        assert loc <= len(inputs)
        output = inputs[loc]

        for module, para_required in zip(self, self._para_required):
            if para_required == 1 and not isinstance(module, (MultiSequential, Tree)):
                input = inputs[loc]
                if isinstance(input, tuple):
                    output = tuple(module(_input) for _input in input)
                else:
                    output = module(input)
            else:
                output = module(*inputs)
            inputs = inputs[:loc] + (output,) + inputs[loc + 1:]
        return output


class MultiSequential(nn.Sequential):
    """Multiple inputs and multiple outputs from multiple mudule"""

    def forward(self, *input):
        for module in self:
            if not isinstance(input, tuple):
                input = (input,)
            input = module(*input)

        if len(input) == 1:
            input, = input
        return input


class Tree(nn.Sequential):
    """Single input and multiple outputs from multiple mudule

    >>> lin1 = nn.Linear(10, 3)
    >>> lin2 = nn.Linear(10, 5)
    >>> t = Tree(lin1, lin2)
    >>> x = torch.randn(10, 10)
    >>> t(x) # outputs (out1, out2) <==> (lin1(x), lin2(x))
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
