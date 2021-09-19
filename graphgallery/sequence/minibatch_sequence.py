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


class SAGEMiniBatchSequence(Sequence):

    def __init__(
        self,
        x,
        y=None,
        out_index=None,
        sizes=[5, 5],
        shuffle=False,
        batch_size=512,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.node_attr, self.adj_matrix, self.batch_nodes = x
        self.y = y
        self.n_batches = int(np.ceil(len(self.batch_nodes) / batch_size))
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.indices = np.arange(len(self.batch_nodes))
        self.sizes = sizes

        self.node_attr = self.astensor(self.node_attr)

    def __len__(self):
        return self.n_batches

    def __getitem__(self, index):
        if self.shuffle:
            idx = self.indices[index *
                               self.batch_size:(index + 1) * self.batch_size]
        else:
            idx = slice(index * self.batch_size, (index + 1) * self.batch_size)

        nodes_input = [self.batch_nodes[idx]]
        for num_sample in self.sizes:
            neighbors = sample_neighbors(
                self.adj_matrix, nodes_input[-1], num_sample).ravel()
            nodes_input.append(neighbors)

        y = self.y[idx] if self.y is not None else None

        return self.astensors([self.node_attr, *nodes_input], y)

    def on_epoch_end(self):
        pass

    def on_epoch_end(self):
        if self.shuffle:
            self._shuffle_batches()

    def _shuffle_batches(self):
        """
         Shuffle all nodes at the end of each epoch
        """
        random.shuffle(self.indices)


def sample_neighbors(adj_matrix, nodes, num_neighbors):
    np.random.shuffle(adj_matrix.T)
    return adj_matrix[nodes, :num_neighbors]
