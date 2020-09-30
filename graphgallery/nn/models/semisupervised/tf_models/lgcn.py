import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery.nn.layers.tf_layers import Top_k_features, LGConvolution, DenseConvolution, Mask
from graphgallery.nn.models import SemiSupervisedModel
from graphgallery.sequence import FullBatchNodeSequence
from graphgallery.utils.decorators import EqualVarLength
from graphgallery import transformers as T


class LGCN(SemiSupervisedModel):
    """
        Implementation of Large-Scale Learnable Graph Convolutional Networks (LGCN).
        `Large-Scale Learnable Graph Convolutional Networks <https://arxiv.org/abs/1808.03965>`
        Tensorflow 1.x implementation: <https://github.com/divelab/lgcn>
    """

    def __init__(self, *graph, adj_transformer="normalize_adj", attr_transformer=None,
                 device='cpu:0', seed=None, name=None, **kwargs):
        """Creat a Large-Scale Learnable Graph Convolutional Networks (LGCN) model.


        This can be instantiated in several ways:

            model = LGCN(graph)
                with a `graphgallery.data.Graph` instance representing
                A sparse, attributed, labeled graph.

            model = LGCN(adj_matrix, attr_matrix, labels)
                where `adj_matrix` is a 2D Scipy sparse matrix denoting the graph,
                 `attr_matrix` is a 2D Numpy array-like matrix denoting the node 
                 attributes, `labels` is a 1D Numpy array denoting the node labels.


        Parameters:
        ----------
        graph: An instance of `graphgallery.data.Graph` or a tuple (list) of inputs.
            A sparse, attributed, labeled graph.
        adj_transformer: string, `transformer`, or None. optional
            How to transform the adjacency matrix. See `graphgallery.transformers`
            (default: :obj:`'normalize_adj'` with normalize rate `-0.5`.
            i.e., math:: \hat{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}}) 
        attr_transformer: string, transformer, or None. optional
            How to transform the node attribute matrix. See `graphgallery.transformers`
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
        kwargs: other customed keyword Parameters.
        """
        super().__init__(*graph, device=device, seed=seed, name=name, **kwargs)

        self.adj_transformer = T.get(adj_transformer)
        self.attr_transformer = T.get(attr_transformer)
        self.process()

    def process_step(self):
        graph = self.graph
        adj_matrix = self.adj_transformer(graph.adj_matrix).toarray()
        attr_matrix = self.attr_transformer(graph.attr_matrix)

        self.feature_inputs, self.structure_inputs = attr_matrix, adj_matrix

    # @EqualVarLength()
    def build(self, hiddens=[32], n_filters=[8, 8], activations=[None, None], dropouts=[0.8, 0.8],
              l2_norms=[5e-4, 5e-4], lr=0.1, use_bias=False, k=8):

        with tf.device(self.device):

            x = Input(batch_shape=[None, self.graph.n_attrs],
                      dtype=self.floatx, name='attr_matrix')
            adj = Input(batch_shape=[None, None],
                        dtype=self.floatx, sparse=False, name='adj_matrix')
            mask = Input(batch_shape=[None], dtype=tf.bool, name='node_mask')

            h = x
            for idx, hid in enumerate(hiddens):
                h = Dropout(rate=dropouts[idx])(h)
                h = DenseConvolution(hid,
                                     use_bias=use_bias,
                                     activation=activations[idx],
                                     kernel_regularizer=regularizers.l2(l2_norms[idx]))([h, adj])

            for idx, n_filter in enumerate(n_filters):
                top_k_h = Top_k_features(k=k)([h, adj])
                cur_h = LGConvolution(n_filter,
                                      kernel_size=k,
                                      use_bias=use_bias,
                                      dropout=dropouts[idx],
                                      activation=activations[idx],
                                      kernel_regularizer=regularizers.l2(l2_norms[idx]))(top_k_h)
                cur_h = BatchNormalization()(cur_h)
                h = Concatenate()([h, cur_h])

            h = Dropout(rate=dropouts[-1])(h)
            h = DenseConvolution(self.graph.n_classes,
                                 use_bias=use_bias,
                                 activation=activations[-1],
                                 kernel_regularizer=regularizers.l2(l2_norms[-1]))([h, adj])

            h = Mask()([h, mask])

            model = Model(inputs=[x, adj, mask], outputs=h)
            model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                          optimizer=Nadam(lr=lr), metrics=['accuracy'])

            self.k = k
            self.model = model

    def train_sequence(self, index, batch_size=np.inf):
        index = T.asintarr(index)
        mask = T.indices2mask(index, self.graph.n_nodes)
        index = get_indice_graph(self.structure_inputs, index, batch_size)
        while index.size < self.k:
            index = get_indice_graph(self.structure_inputs, index)

        structure_inputs = self.structure_inputs[index][:, index]
        feature_inputs = self.feature_inputs[index]
        mask = mask[index]
        labels = self.graph.labels[index[mask]]

        sequence = FullBatchNodeSequence(
            [feature_inputs, structure_inputs, mask], labels, device=self.device)
        return sequence


def get_indice_graph(adj_matrix, indices, size=np.inf, dropout=0.):
    if dropout > 0.:
        indices = np.random.choice(indices, int(
            indices.size * (1 - dropout)), False)
    neighbors = adj_matrix[indices].sum(axis=0).nonzero()[0]
    if neighbors.size > size - indices.size:
        neighbors = np.random.choice(
            list(neighbors), size - len(indices), False)
    indices = np.union1d(indices, neighbors)
    return indices
