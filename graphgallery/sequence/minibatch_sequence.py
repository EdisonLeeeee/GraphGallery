import random
import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from .sequence import Sequence


class MiniBatchSequence(Sequence):

    def __init__(
        self,
        x,
        y,
        out_index=None,
        shuffle=False,
        batch_size=1,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert batch_size == 1
        self.n_batches = len(x)
        self.shuffle = shuffle
        self.indices = list(range(self.n_batches))
        self.batch_size = batch_size
        self.x, self.y, self.out_index = self.astensors(x, y, out_index)

    def __len__(self):
        return self.n_batches

    def __getitem__(self, index):
        idx = self.indices[index]
        return self.x[idx], self.y[idx], self.out_index[idx]

    def on_epoch_end(self):
        if self.shuffle:
            self._shuffle_batches()

    def _shuffle_batches(self):
        """
         Shuffle all nodes at the end of each epoch
        """
        random.shuffle(self.indices)
