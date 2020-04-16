
import numpy as np
import tensorflow as tf

from graphgallery.utils import sample_mask
from .node_sequence import NodeSequence


class NodeSampleSequence(NodeSequence):

    def __init__(
        self,
        inputs, 
        labels,
        neighbors,
        n_sample=100,
        resample=True
    ):

        self.inputs = inputs
        self.labels = labels
        self.n_batches = 1
        self.neighbors = neighbors
        self.n_nodes = inputs[0].shape[0]
        self.n_sample = n_sample
        self.adv_mask = self.smple_nodes()
        self.resample = resample

    def __len__(self):
        return self.n_batches

    def __getitem__(self, index):
        
        item = [*self.inputs, self.adv_mask], self.labels
        if self.resample:
            self.adv_mask = self.smple_nodes()
        return self._to_tensor(item)

    def smple_nodes(self):
        N = self.n_nodes
        flag = np.zeros(N, dtype=np.bool)
        adv_index = np.zeros(self.n_sample, dtype='int32')
        for i in range(self.n_sample):
            n = np.random.randint(0, N)
            while flag[n]:
                n = np.random.randint(0, N)
            adv_index[i] = n
            flag[self.neighbors[n]] = True
            if flag.sum()==N:
                break
        adv_mask = np.zeros(N, dtype='float32')
        adv_mask[adv_index] = 1.
        return adv_mask
