import inspect
import torch.nn as nn


class Sequential(nn.Sequential):
    def __init__(self, *args, inverse=False):
        super().__init__(*args)
        self.inverse = inverse

    def forward(self, *input):
        if self.inverse:
            *others, input = input
        else:
            input, *others = input
        for module in self:
            # TODO: if some modules don't have method `forward`? maybe `__call__`?
            num_paras = len(inspect.signature(module.forward).parameters)
            if num_paras == 1:
                input = module(input)
            else:
                if self.inverse:
                    input = module(*others, input)
                else:
                    input = module(input, *others)
        return input
