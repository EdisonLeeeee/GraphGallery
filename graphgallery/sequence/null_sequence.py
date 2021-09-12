from .base_sequence import Sequence


class NullSequence(Sequence):

    def __init__(self, *dataset, **kwargs):
        super().__init__([dataset], batch_size=None, collate_fn=lambda x: x, **kwargs)
