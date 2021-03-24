import random
import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from .base_sequence import Sequence


class MiniBatchSequence(Sequence):

    def __init__(
        self,
        x,
        y,
        out_weight=None,
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
        self.x, self.y, self.out_weight = self.astensors(x, y, out_weight)

    def __len__(self):
        return self.n_batches

    def __getitem__(self, index):
        idx = self.indices[index]
        return self.x[idx], self.y[idx], self.out_weight[idx]

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
        out_weight=None,
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
        node_attr, adj_matrix = x
        self.y = y
        self.n_batches = int(
            np.ceil(adj_matrix.shape[0] / batch_size)) if batch_size else 1
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.indices = np.arange(adj_matrix.shape[0])
        self.rank = rank
        if rank:
            self.p = column_prop(adj_matrix)

        self.node_attr, self.adj_matrix = node_attr, adj_matrix

    def __len__(self):
        return self.n_batches

    def __getitem__(self, index):
        if not self.batch_size:
            (node_attr, adj_matrix), y = self.full_batch()
        else:
            (node_attr, adj_matrix), y = self.mini_batch(index)

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

            if tf.is_tensor(node_attr):
                node_attr = tf.gather(node_attr, q)
            else:
                node_attr = node_attr[q]

        return self.astensors((node_attr, adj_matrix), y)

    def full_batch(self):
        return (self.node_attr, self.adj_matrix), self.y

    def mini_batch(self, index):
        if self.shuffle:
            idx = self.indices[index *
                               self.batch_size:(index + 1) * self.batch_size]
        else:
            idx = slice(index * self.batch_size, (index + 1) * self.batch_size)

        y = self.y[idx]
        adj_matrix = self.adj_matrix[idx]
        node_attr = self.node_attr

        return (node_attr, adj_matrix), y

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


def column_prop(adj):
    column_norm = sp.linalg.norm(adj, axis=0)
    norm_sum = column_norm.sum()
    return column_norm / norm_sum
