import torch
from torch import nn


class Metric(nn.Module):

    def __init__(self, name=None, dtype=None, **kwargs):
        super().__init__()
        self.name = name
        self.dtype = dtype
        self.reset_states()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def update_state(self):
        raise NotImplementedError

    def reset_states(self):
        raise NotImplementedError

    def result(self):
        raise NotImplementedError
