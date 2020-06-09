import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from graphgallery.nn.layers import GraphEdgeConvolution
from graphgallery.nn.models import SupervisedModel
from graphgallery.sequence import FullBatchNodeSequence
from graphgallery.utils.data_utils import normalize_fn, normalize_adj


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
            adj: shape (N, N), `scipy.sparse.csr_matrix` (or `csc_matrix`) if 
                `is_adj_sparse=True`, `np.array` or `np.matrix` if `is_adj_sparse=False`.
                The input `symmetric` adjacency matrix, where `N` is the number 
                of nodes in graph.
            x: shape (N, F), `scipy.sparse.csr_matrix` (or `csc_matrix`) if 
                `is_x_sparse=True`, `np.array` or `np.matrix` if `is_x_sparse=False`.
                The input node feature matrix, where `F` is the dimension of features.
            labels: `np.array` with shape (N,)
                The ground-truth labels for all nodes in graph.
            norm_adj_rate (Float scalar, optional): 
                The normalize rate for adjacency matrix `adj`. (default: :obj:`-0.5`, 
                i.e., math:: \hat{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}}) 
            norm_x_type (String, optional): 
                How to normalize the node feature matrix. See graphgallery.utils.normalize_fn
                (default :obj: `row_wise`)
            device (String, optional): 
                The device where the model is running on. You can specified `CPU` or `GPU` 
                for the model. (default: :obj: `CPU:0`, i.e., the model is running on 
                the 0-th device `CPU`)
            seed (Positive integer, optional): 
                Used in combination with `tf.random.set_seed` & `np.random.seed` & `random.seed`  
                to create a reproducible sequence of tensors across multiple calls. 
                (default :obj: `None`, i.e., using random seed)
            name (String, optional): 
                Specified name for the model. (default: `class.__name__`)

    """

    def __init__(self, adj, x, labels, norm_adj_rate=-0.5, norm_x_type='row_wise', 
                 device='CPU:0', seed=None, name=None, **kwargs):

        super().__init__(adj, x, labels, device=device, seed=seed, name=name, **kwargs)

        self.norm_adj_rate = norm_adj_rate
        self.norm_x_fn = normalize_fn(norm_x_type)
        self.preprocess(adj, x)

    def preprocess(self, adj, x):
        adj, x = super().preprocess(adj, x)

        if self.norm_adj_rate is not None:
            adj = normalize_adj(adj, self.norm_adj_rate)

        if self.norm_x_fn is not None:
            x = self.norm_x_fn(x)

        with tf.device(self.device):
            adj = adj.tocoo()
            edge_index = np.stack([adj.row, adj.col], axis=1).astype(self.intx, copy=False)
            edge_weight = adj.data.astype(self.floatx, copy=False)
            self.tf_x, self.edge_index, self.edge_weight = self.to_tensor([x, edge_index, edge_weight])

    def build(self, hiddens=[16], activations=['relu'], dropout=0.5,
              lr=0.01, l2_norm=5e-4, use_bias=False):

        assert len(hiddens) == len(activations), "The number of hidden units and " \
                                                "activation function should be the same"

        with tf.device(self.device):
            x = Input(batch_shape=[None, self.n_features], dtype=self.floatx, name='features')
            edge_index = Input(batch_shape=[None, 2], dtype=self.intx, name='edge_index')
            edge_weight = Input(batch_shape=[None],  dtype=self.floatx, name='edge_weight')
            index = Input(batch_shape=[None],  dtype=self.intx, name='index')

            h = x
            for hid, activation in zip(hiddens, activations):
                h = GraphEdgeConvolution(hid, use_bias=use_bias,
                                         activation=activation,
                                         kernel_regularizer=regularizers.l2(l2_norm))([h, edge_index, edge_weight])

                h = Dropout(rate=dropout)(h)

            h = GraphEdgeConvolution(self.n_classes, use_bias=use_bias)([h, edge_index, edge_weight])
            h = tf.gather(h, index)
            output = Softmax()(h)

            model = Model(inputs=[x, edge_index, edge_weight, index], outputs=output)
            model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])

            self.set_model(model)
            self.built = True

    def train_sequence(self, index):
        index = self.to_int(index)
        labels = self.labels[index]
        with tf.device(self.device):
            sequence = FullBatchNodeSequence([self.tf_x, self.edge_index, self.edge_weight, index], labels)
        return sequence

    def predict(self, index):
        super().predict(index)
        index = self.to_int(index)
        with tf.device(self.device):
            logit = self.model.predict_on_batch([self.tf_x, self.edge_index, self.edge_weight, index])

        if tf.is_tensor(logit):
            logit = logit.numpy()
        return logit
    
    def __call__(self, inputs):
        return self.model(inputs)