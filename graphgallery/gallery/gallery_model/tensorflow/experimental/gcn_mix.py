import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from graphgallery.nn.layers.tensorflow import GraphConvolution
from graphgallery.sequence import FullBatchSequence
from ..fastgcn import FastGCN


class GCN_MIX(FastGCN):
    """
        Implementation of Mixed Graph Convolutional Networks (GCN_MIX) 
            occured in FastGCN. 
        GCN_MIX Tensorflow 1.x implementation: <https://github.com/matenure/FastGCN>

    """

    def __init__(self, graph,
                 adj_transform="normalize_adj",
                 attr_transform=None,
                 graph_transform=None,
                 device="cpu", seed=None, name=None, **kwargs):
        """Create Mixed Graph Convolutional Networks (GCN_MIX) occured in FastGCN.

        Calculating `A @ X` in advance to save time.

        This can be instantiated in the following way:

            model = GCN(graph)
                with a `graphgallery.data.Graph` instance representing
                A sparse, attributed, labeled graph.

        Parameters:
        ----------
        graph: An instance of `graphgallery.data.Graph`.
            A sparse, attributed, labeled graph.
        adj_transform: string, `transform`, or None. optional
            How to transform the adjacency matrix. See `graphgallery.functional`
            (default: :obj:`'normalize_adj'` with normalize rate `-0.5`.
            i.e., math:: \hat{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}}) 
        attr_transform: string, `transform`, or None. optional
            How to transform the node attribute matrix. See `graphgallery.functional`
            (default :obj: `None`)
        graph_transform: string, `transform` or None. optional
            How to transform the graph, by default None.
        device: string. optional
            The device where the model is running on. 
            You can specified ``CPU``, ``GPU`` or ``cuda``  
            for the model. (default: :str: `cpu`, i.e., running on the `CPU`)
        seed: interger scalar. optional 
            Used in combination with `tf.random.set_seed` & `np.random.seed` 
            & `random.seed` to create a reproducible sequence of tensors across 
            multiple calls. (default :obj: `None`, i.e., using random seed)
        name: string. optional
            Specified name for the model. (default: :str: `class.__name__`)
        kwargs: other custom keyword parameters.
        """
        super().__init__(graph, device=device, seed=seed, name=name,
                         adj_transform=adj_transform,
                         attr_transform=attr_transform,
                         graph_transform=graph_transform,
                         **kwargs)

    def train_sequence(self, index):
        labels = self.graph.node_label[index]

        sequence = FullBatchSequence(
            [self.cache.X, self.cache.A[index]], labels, device=self.device)
        return sequence
