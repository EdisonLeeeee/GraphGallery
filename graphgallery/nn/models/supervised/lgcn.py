import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Softmax, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras import regularizers

from graphgallery.nn.layers import Top_k_features, LGConvolution, DenseGraphConv
from graphgallery.sequence import FullBatchNodeSequence
from graphgallery.utils.graph_utils import get_indice_graph
from graphgallery.utils.data_utils import sample_mask
from graphgallery.nn.models import SupervisedModel


class LGCN(SupervisedModel):
    """
        Implementation of Large-Scale Learnable Graph Convolutional Networks (LGCN). 
        [Large-Scale Learnable Graph Convolutional Networks](https://arxiv.org/abs/1808.03965)
        Tensorflow 1.x implementation: https://github.com/divelab/lgcn

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
            
        # TODO: the input adj can be dense matrix
#         if sp.isspmatrix(adj):
#             adj = adj.toarray()
            
        if self.is_normalize_x:
            x = self.normalize_x(x)

        self.x_norm, self.adj_norm = x, adj

    def build(self, hiddens=[32], n_filters=[8, 8], activations=[None], dropout=0.8,
              lr=0.1, l2_norm=5e-4, use_bias=False, k=8):

        assert len(hiddens) == len(activations)
        
        with tf.device(self.device):

            x = Input(batch_shape=[None, self.n_features], dtype=self.floatx, name='features')
            adj = Input(batch_shape=[None, None], dtype=self.floatx, sparse=False, name='adj_matrix')
            mask = Input(batch_shape=[None],  dtype=tf.bool, name='mask')

            h = x
            for hid, activation in zip(hiddens, activations):
                h = Dropout(rate=dropout)(h)
                h = DenseGraphConv(hid, use_bias=use_bias, activation=activation,
                                   kernel_regularizer=regularizers.l2(l2_norm))([h, adj])

            for n_filter in n_filters:
                top_k_h = Top_k_features(k=k)([h, adj])
                cur_h = LGConvolution(n_filter, k, use_bias=use_bias,
                                      dropout=dropout, activation=None,
                                      kernel_regularizer=regularizers.l2(l2_norm))(top_k_h)
                cur_h = BatchNormalization()(cur_h)
                h = Concatenate()([h, cur_h])

            h = Dropout(rate=dropout)(h)
            h = DenseGraphConv(self.n_classes, use_bias=use_bias, kernel_regularizer=regularizers.l2(l2_norm))([h, adj])

            h = tf.boolean_mask(h, mask)
            output = Softmax()(h)

            model = Model(inputs=[x, adj, mask], outputs=output)
            model.compile(loss='sparse_categorical_crossentropy', optimizer=Nadam(lr=lr), metrics=['accuracy'])

            self.k = k
            self.set_model(model)
            self.built = True

    def train_sequence(self, index, batch_size=np.inf):
        index = self.to_int(index)
        mask = sample_mask(index, self.n_nodes)
        index = get_indice_graph(self.adj_norm, index, batch_size)
        while index.size < self.k:
            index = get_indice_graph(self.adj_norm, index)
        adj = self.adj_norm[index][:, index].toarray()
        x = self.x_norm[index]
        mask = mask[index]
        labels = self.labels[index[mask]]

        with tf.device(self.device):
            sequence = FullBatchNodeSequence([x, adj, mask], labels)
        return sequence

    def predict(self, index):
        super().predict(index)
        index = self.to_int(index)
        mask = sample_mask(index, self.n_nodes)
        index = get_indice_graph(self.adj_norm, index)
        while index.size < self.k:
            index = get_indice_graph(self.adj_norm, index)
        adj = self.adj_norm[index][:, index].toarray()
        x = self.x_norm[index]
        mask = mask[index]

        with tf.device(self.device):
            x, adj, mask = self.to_tensor([x, adj, mask])
            logit = self.model.predict_on_batch([x, adj, mask])

        if tf.is_tensor(logit):
            logit = logit.numpy()
        return logit
