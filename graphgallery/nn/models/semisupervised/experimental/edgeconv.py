import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery.nn.layers import GraphEdgeConvolution, Gather
from graphgallery.nn.models import SemiSupervisedModel
from graphgallery.sequence import FullBatchNodeSequence
from graphgallery.utils.shape import set_equal_in_length
from graphgallery import astensors, asintarr, normalize_x, normalize_adj, sparse_adj_to_edges, Bunch


class EdgeGCN(SemiSupervisedModel):
    """
        Implementation of Graph Convolutional Networks (GCN) -- Edge Convolution version.
        `Semi-Supervised Classification with Graph Convolutional Networks <https://arxiv.org/abs/1609.02907>`

        Inspired by: tf_geometric and torch_geometric
        tf_geometric: <https://github.com/CrawlScript/tf_geometric>
        torch_geometric: <https://github.com/rusty1s/pytorch_geometric>

        Note:
        ----------
        The Graph Edge Convolutional implements the operation using message passing framework,
            i.e., using Tensor `edge index` and `edge weight` of adjacency matrix to aggregate neighbors'
            message, instead of SparseTensor `adj`.


        Arguments:
        ----------
            adj: shape (N, N), Scipy sparse matrix if  `is_adj_sparse=True`,
                Numpy array-like (or matrix) if `is_adj_sparse=False`.
                The input `symmetric` adjacency matrix, where `N` is the number
                of nodes in graph.
            x: shape (N, F), Scipy sparse matrix if `is_x_sparse=True`,
                Numpy array-like (or matrix) if `is_x_sparse=False`.
                The input node feature matrix, where `F` is the dimension of features.
            labels: Numpy array-like with shape (N,)
                The ground-truth labels for all nodes in graph.
            norm_adj (Float scalar, optional):
                The normalize rate for adjacency matrix `adj`. (default: :obj:`-0.5`,
                i.e., math:: \hat{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}})
            norm_x (String, optional):
                How to normalize the node feature matrix. See `graphgallery.normalize_x`
                (default :obj: `None`)
            device (String, optional):
                The device where the model is running on. You can specified `CPU` or `GPU`
                for the model. (default: :str: `CPU:0`, i.e., running on the 0-th `CPU`)
            seed (Positive integer, optional):
                Used in combination with `tf.random.set_seed` & `np.random.seed` & `random.seed`
                to create a reproducible sequence of tensors across multiple calls.
                (default :obj: `None`, i.e., using random seed)
            name (String, optional):
                Specified name for the model. (default: :str: `class.__name__`)

    """

    def __init__(self, adj, x, labels, norm_adj=-0.5, norm_x=None,
                 device='CPU:0', seed=None, name=None, **kwargs):

        super().__init__(adj, x, labels, device=device, seed=seed, name=name, **kwargs)

        self.norm_adj = norm_adj
        self.norm_x = norm_x
        self.preprocess(adj, x)

    def preprocess(self, adj, x):
        super().preprocess(adj, x)
        adj, x = self.adj, self.x

        if self.norm_adj:
            adj = normalize_adj(adj, self.norm_adj)

        if self.norm_x:
            x = normalize_x(x, norm=self.norm_x)

        with tf.device(self.device):
            edge_index, edge_weight = sparse_adj_to_edges(adj)
            self.x_norm, self.edge_index, self.edge_weight = astensors([x, edge_index, edge_weight])

    def build(self, hiddens=[16], activations=['relu'], dropouts=[0.5], l2_norms=[5e-4],
              lr=0.01, use_bias=False):

        ############# Record paras ###########
        local_paras = locals()
        local_paras.pop('self')
        paras = Bunch(**local_paras)
        hiddens, activations, dropouts, l2_norms = set_equal_in_length(hiddens, activations, dropouts, l2_norms)
        paras.update(Bunch(hiddens=hiddens, activations=activations, dropouts=dropouts, l2_norms=l2_norms))
        # update all parameters
        self.paras.update(paras)
        self.model_paras.update(paras)
        ######################################

        with tf.device(self.device):
            x = Input(batch_shape=[None, self.n_features], dtype=self.floatx, name='features')
            edge_index = Input(batch_shape=[None, 2], dtype=self.intx, name='edge_index')
            edge_weight = Input(batch_shape=[None],  dtype=self.floatx, name='edge_weight')
            index = Input(batch_shape=[None],  dtype=self.intx, name='index')

            h = x
            for hid, activation, dropout, l2_norm in zip(hiddens, activations, dropouts, l2_norms):
                h = GraphEdgeConvolution(hid, use_bias=use_bias,
                                         activation=activation,
                                         kernel_regularizer=regularizers.l2(l2_norm))([h, edge_index, edge_weight])

                h = Dropout(rate=dropout)(h)

            h = GraphEdgeConvolution(self.n_classes, use_bias=use_bias)([h, edge_index, edge_weight])
            h = Gather()([h, index])

            model = Model(inputs=[x, edge_index, edge_weight, index], outputs=h)
            model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                          optimizer=Adam(lr=lr), metrics=['accuracy'])
            self.model = model

    def train_sequence(self, index):
        index = asintarr(index)
        labels = self.labels[index]
        with tf.device(self.device):
            sequence = FullBatchNodeSequence([self.x_norm, self.edge_index, self.edge_weight, index], labels)
        return sequence

    def predict(self, index):
        super().predict(index)
        index = asintarr(index)
        with tf.device(self.device):
            logit = self.model.predict_on_batch([self.x_norm, self.edge_index, self.edge_weight, index])

        if tf.is_tensor(logit):
            logit = logit.numpy()
        return logit

    def __call__(self, inputs):
        return self.model(inputs)
