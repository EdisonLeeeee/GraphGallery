import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from graphgallery.nn.layers import ChebyConvolution
from graphgallery.sequence import FullBatchNodeSequence
from graphgallery.utils.misc import chebyshev_polynomials
from graphgallery.nn.models import SupervisedModel


class ChebyNet(SupervisedModel):
    """
        Implementation of Chebyshev Graph Convolutional Networks (ChebyNet). 
        [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375)
        Tensorflow 1.x implementation: https://github.com/mdeff/cnn_graph, https://github.com/tkipf/gcn
        Keras implementation: https://github.com/aclyde11/ChebyGCN

        Arguments:
        ----------
            adj: `scipy.sparse.csr_matrix` (or `csc_matrix`) with shape (N, N)
                The input `symmetric` adjacency matrix, where `N` is the number of nodes 
                in graph.
            x: `np.array` with shape (N, F)
                The input node feature matrix, where `F` is the dimension of node features.
            labels: `np.array` with shape (N,)
                The ground-truth labels for all nodes in graph.
            order (Positive integer, optional): 
                The order of Chebyshev polynomial filter. (default :obj: `2`)
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

    def __init__(self, adj, x, labels, order=2, normalize_rate=-0.5,
                 is_normalize_x=True, device='CPU:0', seed=None, name=None, **kwargs):

        super().__init__(adj, x, labels,
                         device=device, seed=seed, name=name, **kwargs)

        self.order = order
        self.normalize_rate = normalize_rate
        self.is_normalize_x = is_normalize_x
        self.preprocess(adj, x)

    def preprocess(self, adj, x):
        adj, x = super().preprocess(adj, x)

        if self.normalize_rate is not None:
            adj = chebyshev_polynomials(adj, rate=self.normalize_rate, order=self.order)

        if self.is_normalize_x:
            x = self.normalize_x(x)

        with tf.device(self.device):
            self.tf_x, self.tf_adj = self.to_tensor([x, adj])

    def build(self, hiddens=[32], activations=['relu'], dropout=0.5, lr=0.01, l2_norm=5e-4, ensure_shape=True):

        assert len(hiddens) == len(activations)
        
        with tf.device(self.device):

            x = Input(batch_shape=[None, self.n_features], dtype=self.floatx, name='features')
            adj = [Input(batch_shape=[None, None],
                         dtype=self.floatx, sparse=True, name=f'adj_matrix_{i}') for i in range(self.order+1)]

            index = Input(batch_shape=[None],  dtype=self.intx, name='index')

            h = x
            for hid, activation in zip(hiddens, activations):
                h = ChebyConvolution(hid, order=self.order, activation=activation, 
                                     kernel_regularizer=regularizers.l2(l2_norm))([h, adj])
                h = Dropout(rate=dropout)(h)

            h = ChebyConvolution(self.n_classes, order=self.order)([h, adj])
            if ensure_shape:
                h = tf.ensure_shape(h, [self.n_nodes, self.n_classes])
            h = tf.gather(h, index)
            output = Softmax()(h)

            model = Model(inputs=[x, *adj, index], outputs=output)
            model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])
            self.set_model(model)
            self.built = True

    def train_sequence(self, index):
        index = self.to_int(index)
        labels = self.labels[index]
        with tf.device(self.device):
            sequence = FullBatchNodeSequence([self.tf_x, *self.tf_adj, index], labels)
        return sequence

    def predict(self, index):
        super().predict(index)
        index = self.to_int(index)
        with tf.device(self.device):
            logit = self.model.predict_on_batch([self.tf_x, *self.tf_adj, index])

        if tf.is_tensor(logit):
            logit = logit.numpy()
        return logit
