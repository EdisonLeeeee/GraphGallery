from graphgallery import astensor, astensors
from graphgallery.sequence.base_sequence import Sequence


class FullBatchNodeSequence(Sequence):

    def __init__(self, x, y=None):
        self.x = astensors(x)
        self.y = astensor(y)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.x, self.y

    def on_epoch_end(self):
        pass

    def shuffle(self):
        pass
