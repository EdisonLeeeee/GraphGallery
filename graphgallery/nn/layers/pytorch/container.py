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
            if para_required == 1:
                input = module(input)
            else:
                if self.reverse:
                    input = module(*others, input)
                else:
                    input = module(input, *others)

            if isinstance(input, tuple):
                input, others = split_input(input,
                                            reverse=self.reverse)

        return input


def split_input(input, reverse=True):
    if reverse:
        *others, input = input
    else:
        input, *others = input
    return input, others
