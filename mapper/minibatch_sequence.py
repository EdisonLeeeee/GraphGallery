import random
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from .node_sequence import NodeSequence

class ClusterMiniBatchSequence(NodeSequence):

    def __init__(
        self,
        inputs, 
        labels,
        shuffle_batches=True,
        batch_size=1,
    ):
        assert batch_size == 1
        self.inputs, self.labels = self._to_tensor([inputs, labels])
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
      
    
from graphgallery.utils import sample_neighbors

class SAGEMiniBatchSequence(NodeSequence):

    def __init__(
        self,
        inputs, 
        labels=None,
        adj=None,
        n_samples=[5,5],
        shuffle_batches=False,
        batch_size=512
    ):
        self.features, self.nodes = inputs
        self.labels = labels
        self.adj = adj
        self.n_batches = int(np.ceil(len(self.nodes)/batch_size))
        self.shuffle_batches = shuffle_batches
        self.batch_size = batch_size
        self.indices = np.arange(len(self.nodes))
        self.n_samples = n_samples
        
        self.features = self._to_tensor(self.features)

    def __len__(self):
        return self.n_batches

    def __getitem__(self, index):
        if self.shuffle_batches:
            idx = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        else:
            idx = slice(index*self.batch_size, (index+1)*self.batch_size)
            
        nodes_input = [self.nodes[idx]]
        for n_sample in self.n_samples:
            neighbors = sample_neighbors(self.adj, nodes_input[-1], n_sample).ravel()
            nodes_input.append(neighbors)
            
        labels = self.labels[idx] if self.labels is not None else None

        return self._to_tensor([[self.features, *nodes_input], labels])
    
    def on_epoch_end(self):
        if self.shuffle_batches:
            self.shuffle()
    
    def shuffle(self):
        """
         Shuffle all nodes at the end of each epoch
        """
        random.shuffle(self.indices)
        
        
from graphgallery.utils import column_prop

class FastGCNBatchSequence(NodeSequence):

    def __init__(
        self,
        inputs, 
        labels=None,
        shuffle_batches=False,
        batch_size=None,
        rank=None
    ):
        self.features, self.adj = inputs
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
        if self.batch_size is None:
            (features, adj), labels = self.full_batch(index)
        else:
            (features, adj), labels = self.mini_batch(index)
        
        if self.rank is not None:
            p = self.p
            rank = self.rank
            distr = adj.sum(0).A1.nonzero()[0]
            if rank>distr.size:
                q = distr
            else:
                q = np.random.choice(distr, rank, replace=False, p=p[distr]/p[distr].sum())
            adj = adj[:, q].dot(sp.diags(1.0 / (p[q] * rank)))
            
            if tf.is_tensor(features):
                features = tf.gather(features, q)
            else:
                features = features[q]                
    
        return self._to_tensor([(features, adj), labels])
    
    def full_batch(self, index):
        return (self.features, self.adj), self.labels
        
    def mini_batch(self, index):
        if self.shuffle_batches:
            idx = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        else:
            idx = slice(index*self.batch_size, (index+1)*self.batch_size)
            
        labels = self.labels[idx]
        adj = self.adj[idx]
        features = self.features
            
        return (features, adj), labels

    
    def on_epoch_end(self):
        if self.shuffle_batches:
            self.shuffle()
    
    def shuffle(self):
        """
         Shuffle all nodes at the end of each epoch
        """
        random.shuffle(self.indices)
      
    