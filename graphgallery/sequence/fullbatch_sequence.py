from .base_sequence import Sequence


class FullBatchSequence(Sequence):

    def __init__(self, x, y=None, out_index=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.x = self.astensors(x, device=self.device)
        self.y = self.astensors(y, device=self.device)
        self.out_index = self.astensors(out_index, device=self.device)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.x, self.y, self.out_index

    def on_epoch_begin(self):
        ...

    def on_epoch_end(self):
        ...
