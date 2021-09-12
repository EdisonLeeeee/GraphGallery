from functools import partial

from graphgallery import functional as gf
from torch.utils.data import DataLoader


class Sequence(DataLoader):

    def __init__(self, dataset, device='cpu', escape=None, **kwargs):
        super().__init__(dataset, **kwargs)
        self.astensor = partial(gf.astensor, device=device, escape=escape)
        self.astensors = partial(gf.astensors, device=device, escape=escape)
        self.device = device

    def on_epoch_begin(self):
        ...

    def on_epoch_end(self):
        ...

    def _shuffle_batches(self):
        ...
