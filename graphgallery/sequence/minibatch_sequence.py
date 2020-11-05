import random
import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from graphgallery.sequence.base_sequence import Sequence


class MiniBatchSequence(Sequence):

    def __init__(
        self,
        x,
        y,
        shuffle=False,
        batch_size=1,
        *args, **kwargs
    ):
        """
        Initialize batches.

        Args:
            self: (todo): write your description
            x: (int): write your description
            y: (int): write your description
            shuffle: (bool): write your description
            batch_size: (int): write your description
        """
        super().__init__(*args, **kwargs)
        assert batch_size == 1
        self.x, self.y = self.astensors(x, y)
        self.n_batches = len(self.x)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.indices = list(range(self.n_batches))

    def __len__(self):
        """
        Returns the number of rows in the queue.

        Args:
            self: (todo): write your description
        """
        return self.n_batches

    def __getitem__(self, index):
        """
        Return the item at index

        Args:
            self: (todo): write your description
            index: (int): write your description
        """
        idx = self.indices[index]
        return self.x[idx], self.y[idx]

    def on_epoch_end(self):
        """
        Shuffle the current end of the end of the epoch.

        Args:
            self: (todo): write your description
        """
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
        """
        Initialize the graph.

        Args:
            self: (todo): write your description
            x: (int): write your description
            y: (int): write your description
            n_samples: (int): write your description
            shuffle: (bool): write your description
            batch_size: (int): write your description
        """
        super().__init__(*args, **kwargs)
        self.attr_matrix, self.adj_matrix, self.batch_nodes = x
        self.y = y
        self.n_batches = int(np.ceil(len(self.batch_nodes) / batch_size))
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.indices = np.arange(len(self.batch_nodes))
        self.n_samples = n_samples

        self.attr_matrix = self.astensor(self.attr_matrix)

    def __len__(self):
        """
        Returns the number of rows in the queue.

        Args:
            self: (todo): write your description
        """
        return self.n_batches

    def __getitem__(self, index):
        """
        Returns the item at index.

        Args:
            self: (todo): write your description
            index: (int): write your description
        """
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

        return self.astensors([self.attr_matrix, *nodes_input], y)

    def on_epoch_end(self):
        """
        Shuffle the current end of the end of the epoch.

        Args:
            self: (todo): write your description
        """
        if self.shuffle:
            self._shuffle_batches()

    def _shuffle_batches(self):
        """
         Shuffle all nodes at the end of each epoch
        """
        random.shuffle(self.indices)


def sample_neighbors(adj_matrix, nodes, n_neighbors):
    """
    Returns a random neighbors of neighbors from the graph with a given adjacency matrix.

    Args:
        adj_matrix: (array): write your description
        nodes: (todo): write your description
        n_neighbors: (int): write your description
    """
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
        """
        Initialize the graph.

        Args:
            self: (todo): write your description
            x: (int): write your description
            y: (int): write your description
            shuffle: (bool): write your description
            batch_size: (int): write your description
            rank: (int): write your description
        """
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
        """
        Returns the number of rows in the queue.

        Args:
            self: (todo): write your description
        """
        return self.n_batches

    def __getitem__(self, index):
        """
        Get a tensor.

        Args:
            self: (todo): write your description
            index: (int): write your description
        """
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

        return self.astensors((attr_matrix, adj_matrix), y)

    def full_batch(self):
        """
        Returns the adjacency matrix.

        Args:
            self: (todo): write your description
        """
        return (self.attr_matrix, self.adj_matrix), self.y

    def mini_batch(self, index):
        """
        Create a batch of the matrix.

        Args:
            self: (todo): write your description
            index: (int): write your description
        """
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
        """
        Shuffle the current end of the end of the epoch.

        Args:
            self: (todo): write your description
        """
        if self.shuffle:
            self._shuffle_batches()

    def _shuffle_batches(self):
        """
         Shuffle all nodes at the end of each epoch
        """
        random.shuffle(self.indices)


def column_prop(adj):
    """
    Return the column properties of the adjac.

    Args:
        adj: (todo): write your description
    """
    column_norm = sp.linalg.norm(adj, axis=0)
    norm_sum = column_norm.sum()
    return column_norm / norm_sum
