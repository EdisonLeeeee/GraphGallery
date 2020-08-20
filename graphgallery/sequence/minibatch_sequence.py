import random
import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from graphgallery.sequence.node_sequence import NodeSequence
from graphgallery.utils.misc import column_prop
from graphgallery.utils.graph import sample_neighbors
from graphgallery import astensors


class ClusterMiniBatchSequence(NodeSequence):

    def __init__(
        self,
        inputs,
        labels,
        shuffle_batches=True,
        batch_size=1,
    ):
        assert batch_size == 1
        self.inputs, self.labels = astensors([inputs, labels])
        self.n_batches = len(self.inputs)
        self.shuffle_batches = shuffle_batches
        self.batch_size = batch_size
        self.indices = list(range(self.n_batches))

    def __len__(self):
        return self.n_batches

    def __getitem__(self, index):
        idx = self.indices[index]
        labels = self.labels[idx]
        return self.inputs[idx], labels

    def on_epoch_end(self):
        if self.shuffle_batches:
            self.shuffle()

    def shuffle(self):
        """
         Shuffle all nodes at the end of each epoch
        """
        random.shuffle(self.indices)


class SAGEMiniBatchSequence(NodeSequence):

    def __init__(
        self,
        inputs,
        labels=None,
        neighbors=None,
        n_samples=[5, 5],
        shuffle_batches=False,
        batch_size=512
    ):
        self.x, self.nodes = inputs
        self.labels = labels
        self.neighbors = neighbors
        self.n_batches = int(np.ceil(len(self.nodes)/batch_size))
        self.shuffle_batches = shuffle_batches
        self.batch_size = batch_size
        self.indices = np.arange(len(self.nodes))
        self.n_samples = n_samples

        self.x = astensors(self.x)

    def __len__(self):
        return self.n_batches

    def __getitem__(self, index):
        if self.shuffle_batches:
            idx = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        else:
            idx = slice(index*self.batch_size, (index+1)*self.batch_size)

        nodes_input = [self.nodes[idx]]
        for n_sample in self.n_samples:
            neighbors = sample_neighbors(self.neighbors, nodes_input[-1], n_sample).ravel()
            nodes_input.append(neighbors)

        labels = self.labels[idx] if self.labels is not None else None

        return astensors([[self.x, *nodes_input], labels])

    def on_epoch_end(self):
        if self.shuffle_batches:
            self.shuffle()

    def shuffle(self):
        """
         Shuffle all nodes at the end of each epoch
        """
        random.shuffle(self.indices)


class FastGCNBatchSequence(NodeSequence):

    def __init__(
        self,
        inputs,
        labels=None,
        shuffle_batches=False,
        batch_size=None,
        rank=None
    ):
        self.x, self.adj = inputs
        self.labels = labels
        self.n_batches = int(np.ceil(self.adj.shape[0]/batch_size)) if batch_size else 1
        self.shuffle_batches = shuffle_batches
        self.batch_size = batch_size
        self.indices = np.arange(self.adj.shape[0])
        self.rank = rank
        if rank:
            self.p = column_prop(self.adj)

    def __len__(self):
        return self.n_batches

    def __getitem__(self, index):
        if not self.batch_size:
            (x, adj), labels = self.full_batch(index)
        else:
            (x, adj), labels = self.mini_batch(index)

        if self.rank:
            p = self.p
            rank = self.rank
            distr = adj.sum(0).A1.nonzero()[0]
            if rank > distr.size:
                q = distr
            else:
                q = np.random.choice(distr, rank, replace=False, p=p[distr]/p[distr].sum())
            adj = adj[:, q].dot(sp.diags(1.0 / (p[q] * rank)))

            if tf.is_tensor(x):
                x = tf.gather(x, q)
            else:
                x = x[q]

        return astensors([(x, adj), labels])

    def full_batch(self, index):
        return (self.x, self.adj), self.labels

    def mini_batch(self, index):
        if self.shuffle_batches:
            idx = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        else:
            idx = slice(index*self.batch_size, (index+1)*self.batch_size)

        labels = self.labels[idx]
        adj = self.adj[idx]
        features = self.x

        return (features, adj), labels

    def on_epoch_end(self):
        if self.shuffle_batches:
            self.shuffle()

    def shuffle(self):
        """
         Shuffle all nodes at the end of each epoch
        """
        random.shuffle(self.indices)
