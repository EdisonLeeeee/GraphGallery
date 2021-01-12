from .base_sequence import Sequence


class NullSequence(Sequence):

    def __init__(self, x, y=None, out_weight=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.x = x
        self.y = y
        self.out_weight = out_weight

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.x, self.y, self.out_weight

    def on_epoch_begin(self):
        ...

    def on_epoch_end(self):
        ...

    def _shuffle_batches(self):
        ...
