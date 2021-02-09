import inspect
import torch.nn as nn


class Sequential(nn.Sequential):
    def forward(self, *input):
        input, *others = input
        for module in self:
            # TODO: if some modules don't have method `forward`? maybe `__call__`?
            num_paras = len(inspect.signature(module.forward).parameters)
            if num_paras == 1:
                input = module(input)
            else:
                input = module(input, *others)
        return input
