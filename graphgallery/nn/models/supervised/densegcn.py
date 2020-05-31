import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from graphgallery.nn.layers import DenseGraphConv
from graphgallery.sequence import FullBatchNodeSequence
from graphgallery.nn.models import SupervisedModel


class DenseGCN(SupervisedModel):
    """
        Implementation of Graph Convolutional Networks (GCN). 
        [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)
        Tensorflow 1.x implementation: https://github.com/tkipf/gcn
        Pytorch implementation: https://github.com/tkipf/pygcn

        Note:
        ----------
        The input adjacency matrix will be transformed into dense matrix, 
        which will causes a lot of comsuming of memory. It's not recommended 
        to use this model in large dataset.

        Arguments:
        ----------
            adj: `scipy.sparse.csr_matrix` (or `csc_matrix`) with shape (N, N)
                The input `symmetric` adjacency matrix, where `N` is the number of nodes 
                in graph.
            x: `np.array` with shape (N, F)
                The input node feature matrix, where `F` is the dimension of node features.
            labels: `np.array` with shape (N,)
                The ground-truth labels for all nodes in graph.
            normalize_rate (Float scalar, optional): 
                The normalize rate for adjacency matrix `adj`. (default: :obj:`-0.5`, 
                i.e., math:: \hat{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}}) 
            is_normalize_x (Boolean, optional): 
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

    def __init__(self, adj, x, labels, normalize_rate=-0.5, is_normalize_x=True, 
                 device='CPU:0', seed=None, name=None, **kwargs):

        super().__init__(adj, x, labels, device=device, seed=seed, name=name, **kwargs)

        self.normalize_rate = normalize_rate
        self.is_normalize_x = is_normalize_x
        self.preprocess(adj, x)
        # set to `False` to suggest the Dense inputs
        self.sparse = False

    def preprocess(self, adj, x):
        adj, x = super().preprocess(adj, x)

        if self.normalize_rate is not None:
            adj = self.normalize_adj(adj, self.normalize_rate)

        if sp.isspmatrix(adj):
            adj = adj.toarray()

        if self.is_normalize_x:
            x = self.normalize_x(x)

        with tf.device(self.device):
            self.tf_x, self.tf_adj = self.to_tensor([x, adj])

    def build(self, hiddens=[16], activations=['relu'], dropout=0.5,
              lr=0.01, l2_norm=5e-4, use_bias=False):
        
        assert len(hiddens) == len(activations)

        with tf.device(self.device):

            x = Input(batch_shape=[self.n_nodes, self.n_features], dtype=self.floatx, name='features')
            adj = Input(batch_shape=[self.n_nodes, self.n_nodes], dtype=self.floatx, name='adj_matrix')
            index = Input(batch_shape=[None],  dtype=self.intx, name='index')

            h = x
            for hid, activation in zip(hiddens, activations):
                h = DenseGraphConv(hid, use_bias=use_bias,
                                   activation=activation,
                                   kernel_regularizer=regularizers.l2(l2_norm))([h, adj])

                h = Dropout(rate=dropout)(h)

            h = DenseGraphConv(self.n_classes, use_bias=use_bias)([h, adj])
            h = tf.ensure_shape(h, [self.n_nodes, self.n_classes])
            h = tf.gather(h, index)
            output = Softmax()(h)

            model = Model(inputs=[x, adj, index], outputs=output)
            model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])

            self._model = model
            self.built = True

    def train_sequence(self, index):
        index = self.to_int(index)
        labels = self.labels[index]
        with tf.device(self.device):
            sequence = FullBatchNodeSequence([self.tf_x, self.tf_adj, index], labels)
        return sequence

    def predict(self, index):
        super().predict(index)
        index = self.to_int(index)
        with tf.device(self.device):
            index = self.to_tensor(index)
            logit = self.model.predict_on_batch([self.tf_x, self.tf_adj, index])

        if tf.is_tensor(logit):
            logit = logit.numpy()
        return logit
