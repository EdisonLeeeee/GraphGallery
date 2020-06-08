import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from graphgallery.nn.layers import GraphAttention
from graphgallery.sequence import FullBatchNodeSequence
from graphgallery.nn.models import SupervisedModel


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

    def __init__(self, adj, x, labels, normalize_rate=None, is_normalize_x=True, 
                 device='CPU:0', seed=None, name=None, **kwargs):

        super().__init__(adj, x, labels, device=device, seed=seed, name=name, **kwargs)

        self.normalize_rate = normalize_rate
        self.is_normalize_x = is_normalize_x
        self.preprocess(adj, x)

    def preprocess(self, adj, x):
        adj, x = super().preprocess(adj, x)

        if self.normalize_rate is None:
            adj = adj + sp.eye(adj.shape[0])
        else:
            adj = self.normalize_adj(adj, self.normalize_rate)

        if self.is_normalize_x:
            x = self.normalize_x(x)

        with tf.device(self.device):
            self.tf_x, self.adj_norm = self.to_tensor([x, adj])

    def build(self, hiddens=[8], n_heads=[8], activations=['elu'], dropout=0.6, lr=0.01, l2_norm=5e-4, ensure_shape=True):
        
        assert len(hiddens) == len(activations) == len(n_heads)
        
        with tf.device(self.device):

            x = Input(batch_shape=[None, self.n_features], dtype=self.floatx, name='features')
            adj = Input(batch_shape=[None, None], dtype=self.floatx, sparse=True, name='adj_matrix')
            index = Input(batch_shape=[None],  dtype=self.intx, name='index')

            h = x
            for hid, n_head, activation in zip(hiddens, n_heads, activations):
                h = GraphAttention(hid, attn_heads=n_head,
                                   attn_heads_reduction='concat',
                                   activation=activation,
                                   kernel_regularizer=regularizers.l2(l2_norm),
                                   attn_kernel_regularizer=regularizers.l2(l2_norm),
                                   )([h, adj])
                h = Dropout(rate=dropout)(h)

            h = GraphAttention(self.n_classes, attn_heads=1, attn_heads_reduction='average')([h, adj])
            if ensure_shape:
                h = tf.ensure_shape(h, [self.n_nodes, self.n_classes])
            h = tf.gather(h, index)
            output = Softmax()(h)

            model = Model(inputs=[x, adj, index], outputs=output)
            model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])

            self.set_model(model)
            self.built = True

    def train_sequence(self, index):
        index = self.to_int(index)
        labels = self.labels[index]
        with tf.device(self.device):
            sequence = FullBatchNodeSequence([self.tf_x, self.adj_norm, index], labels)
        return sequence

    def predict(self, index):
        super().predict(index)
        index = self.to_int(index)
        with tf.device(self.device):
            index = self.to_tensor(index)
            logit = self.model.predict_on_batch([self.tf_x, self.adj_norm, index])

        if tf.is_tensor(logit):
            logit = logit.numpy()
        return logit
