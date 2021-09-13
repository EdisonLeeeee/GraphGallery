from .base_sequence import Sequence
from graphgallery import functional as gf


class FullBatchSequence(Sequence):

    def __init__(self, x, y=None, out_index=None, device='cpu', escape=None, **kwargs):
        dataset = gf.astensors(x, y, out_index, device=device, escape=escape)
        super().__init__([dataset], batch_size=None, collate_fn=lambda x: x, device=device, escape=escape, **kwargs)
