from graphgallery.sequence import FullBatchSequence
from graphgallery.gallery.nodeclas import TensorFlow
from ..fastgcn import FastGCN


@TensorFlow.register()
class GCN_MIX(FastGCN):
    """
        Implementation of Mixed Graph Convolutional Networks (GCN_MIX) 
            occured in FastGCN. 
        GCN_MIX Tensorflow 1.x implementation: <https://github.com/matenure/FastGCN>

    """

    def train_sequence(self, index):
        labels = self.graph.node_label[index]

        sequence = FullBatchSequence(
            [self.cache.X, self.cache.A[index]], labels, device=self.device)
        return sequence
