
import numpy as np
import tensorflow as tf

from graphgallery.sequence.base_sequence import Sequence


class SBVATSampleSequence(Sequence):

    def __init__(
        self,
        x,
        y,
        neighbors,
        n_samples=50,
        resample=True,
        *args, **kwargs
    ):
        """
        Initialize the graph.

        Args:
            self: (todo): write your description
            x: (int): write your description
            y: (int): write your description
            neighbors: (int): write your description
            n_samples: (int): write your description
            resample: (int): write your description
        """
        super().__init__(*args, **kwargs)
        self.x = x
        self.y = y
        self.neighbors = neighbors
        self.n_nodes = x[0].shape[0]
        self.n_samples = n_samples
        self.adv_mask = self.smple_nodes()
        self.resample = resample

    def __len__(self):
        """
        Returns the number of rows

        Args:
            self: (todo): write your description
        """
        return 1

    def __getitem__(self, index):
        """
        Get the index of the item

        Args:
            self: (todo): write your description
            index: (int): write your description
        """
        return self.astensors(*self.x, self.adv_mask), self.astensor(self.y)

    def on_epoch_end(self):
        """
        Resample on the mask

        Args:
            self: (todo): write your description
        """
        if self.resample:
            self.adv_mask = self.smple_nodes()

    def smple_nodes(self):
        """
        Smplements n_n_mask.

        Args:
            self: (todo): write your description
        """
        N = self.n_nodes
        flag = np.zeros(N, dtype=np.bool)
        adv_index = np.zeros(self.n_samples, dtype='int32')
        for i in range(self.n_samples):
            n = np.random.randint(0, N)
            while flag[n]:
                n = np.random.randint(0, N)
            adv_index[i] = n
            flag[self.neighbors[n]] = True
            if flag.sum() == N:
                break
        adv_mask = np.zeros(N, dtype='float32')
        adv_mask[adv_index] = 1.
        return adv_mask
