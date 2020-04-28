import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from graphgallery.nn.layers import GraphAttention
from graphgallery.mapper import FullBatchNodeSequence
from .base import SupervisedModel


class GAT(SupervisedModel):
    """
        Implementation of Graph Attention Networks (GAT). 
        [Graph Attention Networks](https://arxiv.org/abs/1710.10903)
        Tensorflow 1.x implementation: https://github.com/PetarV-/GAT
        Pytorch implementation: https://github.com/Diego999/pyGAT
        Keras implementation: https://github.com/danielegrattarola/keras-gat

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

    def __init__(self, adj, features, labels, normalize_rate=None, normalize_features=True, device='CPU:0', seed=None, **kwargs):

        super().__init__(adj, features, labels, device=device, seed=seed, **kwargs)

        self.normalize_rate = normalize_rate
        self.normalize_features = normalize_features
        self.preprocess(adj, features)

    def preprocess(self, adj, features):

        if self.normalize_rate is None:
            adj = adj + sp.eye(adj.shape[0])
        else:
            adj = self._normalize_adj(adj, self.normalize_rate)

        if self.normalize_features:
            features = self._normalize_features(features)

        with tf.device(self.device):
            self.features, self.adj = self._to_tensor([features, adj])

    def build(self, hidden_layers=[8], n_heads=[8], activations=['elu'], dropout=0.6, learning_rate=0.01, l2_norm=5e-4):

        with tf.device(self.device):

            x = Input(batch_shape=[self.n_nodes, self.n_features], dtype=tf.float32, name='features')
            adj = Input(batch_shape=[self.n_nodes, self.n_nodes], dtype=tf.float32, sparse=True, name='adj_matrix')
            index = Input(batch_shape=[None],  dtype=tf.int32, name='index')

            h = x
            for hid, n_head, activation in zip(hidden_layers, n_heads, activations):
                h = GraphAttention(hid, attn_heads=n_head,
                                   attn_heads_reduction='concat',
                                   activation=activation,
                                   kernel_regularizer=regularizers.l2(l2_norm),
                                   attn_kernel_regularizer=regularizers.l2(l2_norm),
                                   )([h, adj])
                h = Dropout(rate=dropout)(h)

            h = GraphAttention(self.n_classes, attn_heads=1, attn_heads_reduction='average')([h, adj])
            h = tf.ensure_shape(h, [self.n_nodes, self.n_classes])
            h = tf.gather(h, index)
            output = Softmax()(h)

            model = Model(inputs=[x, adj, index], outputs=output)
            model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])

            self.model = model
            self.built = True

    def train_sequence(self, index):
        index = self._check_and_convert(index)
        labels = self.labels[index]
        with tf.device(self.device):
            sequence = FullBatchNodeSequence([self.features, self.adj, index], labels)
        return sequence

    def predict(self, index):
        super().predict(index)
        index = self._check_and_convert(index)
        with tf.device(self.device):
            index = self._to_tensor(index)
            logit = self.model.predict_on_batch([self.features, self.adj, index])

        return logit.numpy()
