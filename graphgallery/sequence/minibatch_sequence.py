import random
import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from graphgallery import astensor, astensors
from graphgallery.sequence.base_sequence import Sequence


class ClusterMiniBatchSequence(Sequence):

    def __init__(
        self,
        x,
        y,
        shuffle=False,
        batch_size=1,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert batch_size == 1
        self.x, self.y = astensors(x, y)
        self.n_batches = len(self.x)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.indices = list(range(self.n_batches))

    def __len__(self):
        return self.n_batches

    def __getitem__(self, index):
        idx = self.indices[index]
        return self.x[idx], self.y[idx]

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
        n_samples=[5, 5],
        shuffle=False,
        batch_size=512,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.attr_matrix, self.adj_matrix, self.batch_nodes = x
        self.y = y
        self.n_batches = int(np.ceil(len(self.batch_nodes) / batch_size))
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.indices = np.arange(len(self.batch_nodes))
        self.n_samples = n_samples

        self.attr_matrix = astensor(self.attr_matrix)

    def __len__(self):
        return self.n_batches

    def __getitem__(self, index):
        if self.shuffle:
            idx = self.indices[index *
                               self.batch_size:(index + 1) * self.batch_size]
        else:
            idx = slice(index * self.batch_size, (index + 1) * self.batch_size)

        nodes_input = [self.batch_nodes[idx]]
        for n_sample in self.n_samples:
            neighbors = sample_neighbors(
                self.adj_matrix, nodes_input[-1], n_sample).ravel()
            nodes_input.append(neighbors)

        y = self.y[idx] if self.y is not None else None

        return astensors([self.attr_matrix, *nodes_input], y)

    def on_epoch_end(self):
        if self.shuffle:
            self._shuffle_batches()

    def _shuffle_batches(self):
        """
         Shuffle all nodes at the end of each epoch
        """
        random.shuffle(self.indices)


def sample_neighbors(adj_matrix, nodes, n_neighbors):
    np.random.shuffle(adj_matrix.T)
    return adj_matrix[nodes, :n_neighbors]


class FastGCNBatchSequence(Sequence):

    def __init__(
        self,
        x,
        y=None,
        shuffle=False,
        batch_size=None,
        rank=None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        attr_matrix, adj_matrix = x
        self.y = y
        self.n_batches = int(
            np.ceil(adj_matrix.shape[0] / batch_size)) if batch_size else 1
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.indices = np.arange(adj_matrix.shape[0])
        self.rank = rank
        if rank:
            self.p = column_prop(adj_matrix)

        self.attr_matrix, self.adj_matrix = attr_matrix, adj_matrix

    def __len__(self):
        return self.n_batches

    def __getitem__(self, index):
        if not self.batch_size:
            (attr_matrix, adj_matrix), y = self.full_batch()
        else:
            (attr_matrix, adj_matrix), y = self.mini_batch(index)

        if self.rank:
            p = self.p
            rank = self.rank
            distr = adj_matrix.sum(0).A1.nonzero()[0]
            if rank > distr.size:
                q = distr
            else:
                q = np.random.choice(
                    distr, rank, replace=False, p=p[distr] / p[distr].sum())
            adj_matrix = adj_matrix[:, q].dot(sp.diags(1.0 / (p[q] * rank)))

            if tf.is_tensor(attr_matrix):
                attr_matrix = tf.gather(attr_matrix, q)
            else:
                attr_matrix = attr_matrix[q]

        return astensors((attr_matrix, adj_matrix), y)

    def full_batch(self):
        return (self.attr_matrix, self.adj_matrix), self.y

    def mini_batch(self, index):
        if self.shuffle:
            idx = self.indices[index *
                               self.batch_size:(index + 1) * self.batch_size]
        else:
            idx = slice(index * self.batch_size, (index + 1) * self.batch_size)

        y = self.y[idx]
        adj_matrix = self.adj_matrix[idx]
        attr_matrix = self.attr_matrix

        return (attr_matrix, adj_matrix), y

    def on_epoch_end(self):
        if self.shuffle:
            self._shuffle_batches()

    def _shuffle_batches(self):
        """
         Shuffle all nodes at the end of each epoch
        """
        random.shuffle(self.indices)


def column_prop(adj):
    column_norm = sp.linalg.norm(adj, axis=0)
    norm_sum = column_norm.sum()
    return column_norm / norm_sum
