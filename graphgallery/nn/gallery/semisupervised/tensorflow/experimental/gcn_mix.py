import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from graphgallery.nn.layers.tensorflow import GraphConvolution
from graphgallery.sequence import FullBatchNodeSequence
from ..fastgcn import FastGCN


class GCN_MIX(FastGCN):
    """
        Implementation of Mixed Graph Convolutional Networks (GCN_MIX) 
            occured in FastGCN. 
        GCN_MIX Tensorflow 1.x implementation: <https://github.com/matenure/FastGCN>

    """

    def __init__(self, *graph, adj_transform="normalize_adj", attr_transform=None,
                 device='cpu:0', seed=None, name=None, **kwargs):
        """Create Mixed Graph Convolutional Networks (GCN_MIX) occured in FastGCN.

        Calculating `A @ X` in advance to save time.


        This can be instantiated in several ways:

            model = GCN(graph)
                with a `graphgallery.data.Graph` instance representing
                A sparse, attributed, labeled graph.

            model = GCN(adj_matrix, attr_matrix, labels)
                where `adj_matrix` is a 2D Scipy sparse matrix denoting the graph,
                 `attr_matrix` is a 2D Numpy array-like matrix denoting the node 
                 attributes, `labels` is a 1D Numpy array denoting the node labels.


        Parameters:
        ----------
        graph: An instance of `graphgallery.data.Graph` or a tuple (list) of inputs.
            A sparse, attributed, labeled graph.
        adj_transform: string, `transform`, or None. optional
            How to transform the adjacency matrix. See `graphgallery.functional`
            (default: :obj:`'normalize_adj'` with normalize rate `-0.5`.
            i.e., math:: \hat{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}}) 
        attr_transform: string, `transform`, or None. optional
            How to transform the node attribute matrix. See `graphgallery.functional`
            (default :obj: `None`)
        device: string. optional 
            The device where the model is running on. You can specified `CPU` or `GPU` 
            for the model. (default: :str: `CPU:0`, i.e., running on the 0-th `CPU`)
        seed: interger scalar. optional 
            Used in combination with `tf.random.set_seed` & `np.random.seed` 
            & `random.seed` to create a reproducible sequence of tensors across 
            multiple calls. (default :obj: `None`, i.e., using random seed)
        name: string. optional
            Specified name for the model. (default: :str: `class.__name__`)
        kwargs: other custom keyword parameters.
        """
        super().__init__(*graph,
                         adj_transform=adj_transform, attr_transform=attr_transform,
                         device=device, seed=seed, name=name, **kwargs)

    def train_sequence(self, index):
        labels = self.graph.labels[index]

        sequence = FullBatchNodeSequence(
            [self.feature_inputs, self.structure_inputs[index]], labels, device=self.device)
        return sequence
