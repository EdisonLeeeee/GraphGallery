import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from graphgallery.nn.layers import GraphEdgeConvolution
from graphgallery.mapper import FullBatchNodeSequence
from graphgallery.nn.models.base import SupervisedModel


class EdgeGCN(SupervisedModel):
    """
        Implementation of Graph Convolutional Networks (GCN) -- Edge Convolution version. 
        [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)

        Inspired by: tf_geometric and torch_geometric
        tf_geometric: https://github.com/CrawlScript/tf_geometric
        torch_geometric: https://github.com/rusty1s/pytorch_geometric

        Note:
        ----------
        The Graph Edge Convolutional implements the operation using message passing framework,
            i.e., using Tensor `edge index` and `edge weight` of adjacency matrix to aggregate neighbors'
            message, instead of SparseTensor `adj`.


        Arguments:
        ----------
            adj: `scipy.sparse.csr_matrix` (or `csc_matrix`) with shape (N, N)
                The input `symmetric` adjacency matrix, where `N` is the number of nodes 
                in graph.
            features: `np.array` with shape (N, F)
                The input node feature matrix, where `F` is the dimension of node features.
            labels: `np.array` with shape (N,)
                The ground-truth labels for all nodes in graph.
            normalize_rate (Float scalar, optional): 
                The normalize rate for adjacency matrix `adj`. (default: :obj:`-0.5`, 
                i.e., math:: \hat{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}}) 
            normalize_features (Boolean, optional): 
                Whether to use row-normalize for node feature matrix. 
                (default :obj: `True`)
            device (String, optional): 
                The device where the model is running on. You can specified `CPU` or `GPU` 
                for the model. (default: :obj: `CPU:0`, i.e., the model is running on 
                the 0-th device `CPU`)
            seed (Positive integer, optional): 
                Used in combination with `tf.random.set_seed & np.random.seed & random.seed` 
                to create a reproducible sequence of tensors across multiple calls. 
                (default :obj: `None`, i.e., using random seed)
            name (String, optional): 
                Name for the model. (default: name of class)

    """

    def __init__(self, adj, features, labels, normalize_rate=-0.5, normalize_features=True, device='CPU:0', seed=None, **kwargs):

        super().__init__(adj, features, labels, device=device, seed=seed, **kwargs)

        self.normalize_rate = normalize_rate
        self.normalize_features = normalize_features
        self.preprocess(adj, features)

    def preprocess(self, adj, features):
        super().preprocess(adj, features)

        if self.normalize_rate is not None:
            adj = self._normalize_adj(adj, self.normalize_rate)

        if self.normalize_features:
            features = self._normalize_features(features)

        with tf.device(self.device):
            adj = adj.tocoo()
            edge_index = np.stack([adj.row, adj.col], axis=1).astype('int64', copy=False)
            edge_weight = adj.data.astype('float32', copy=False)
            self.features, self.edge_index, self.edge_weight = self._to_tensor([features, edge_index, edge_weight])
            
    def build(self, hidden_layers=[16], activations=['relu'], dropout=0.5,
              learning_rate=0.01, l2_norm=5e-4, use_bias=False):

        with tf.device(self.device):
            x = Input(batch_shape=[None, self.n_features], dtype=tf.float32, name='features')
            edge_index = Input(batch_shape=[None, 2], dtype=tf.int64, name='edge_index')
            edge_weight = Input(batch_shape=[None],  dtype=tf.float32, name='edge_weight')
            index = Input(batch_shape=[None],  dtype=tf.int64, name='index')

            h = x
            for hid, activation in zip(hidden_layers, activations):
                h = GraphEdgeConvolution(hid, use_bias=use_bias,
                                         activation=activation,
                                         kernel_regularizer=regularizers.l2(l2_norm))([h, edge_index, edge_weight])

                h = Dropout(rate=dropout)(h)

            h = GraphEdgeConvolution(self.n_classes, use_bias=use_bias)([h, edge_index, edge_weight])
            h = tf.gather(h, index)
            output = Softmax()(h)

            model = Model(inputs=[x, edge_index, edge_weight, index], outputs=output)
            model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])

            self.model = model
            self.built = True

    def train_sequence(self, index):
        index = self._check_and_convert(index)
        labels = self.labels[index]
        with tf.device(self.device):
            sequence = FullBatchNodeSequence([self.features, self.edge_index, self.edge_weight, index], labels)
        return sequence

    def predict(self, index):
        super().predict(index)
        index = self._check_and_convert(index)
        with tf.device(self.device):
            index = self._to_tensor(index)
            logit = self.model.predict_on_batch([self.features, self.edge_index, self.edge_weight, index])

        if tf.is_tensor(logit):
            logit = logit.numpy()
        return logit
