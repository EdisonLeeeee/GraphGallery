import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery.nn.layers import ChebyConvolution, Gather
from graphgallery.sequence import FullBatchNodeSequence
from graphgallery.nn.models import SemiSupervisedModel
from graphgallery.utils.misc import chebyshev_polynomials
from graphgallery.utils.shape import set_equal_in_length
from graphgallery import astensors, asintarr, normalize_x, Bunch


class ChebyNet(SemiSupervisedModel):
    """
        Implementation of Chebyshev Graph Convolutional Networks (ChebyNet).
        `Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering <https://arxiv.org/abs/1606.09375>`
        Tensorflow 1.x implementation: <https://github.com/mdeff/cnn_graph>, <https://github.com/tkipf/gcn>
        Keras implementation: <https://github.com/aclyde11/ChebyGCN>

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
            order (Positive integer, optional):
                The order of Chebyshev polynomial filter. (default :obj: `2`)
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

    def __init__(self, adj, x, labels, order=2, norm_adj=-0.5,
                 norm_x=None, device='CPU:0', seed=None, name=None, **kwargs):

        super().__init__(adj, x, labels,
                         device=device, seed=seed, name=name, **kwargs)

        self.order = order
        self.norm_adj = norm_adj
        self.norm_x = norm_x
        self.preprocess(adj, x)

    def preprocess(self, adj, x):
        super().preprocess(adj, x)
        adj, x = self.adj, self.x

        if self.norm_adj:
            adj = chebyshev_polynomials(adj, rate=self.norm_adj, order=self.order)

        if self.norm_x:
            x = normalize_x(x, norm=self.norm_x)

        with tf.device(self.device):
            self.x_norm, self.adj_norm = astensors([x, adj])

    def build(self, hiddens=[16], activations=['relu'], dropouts=[0.5], l2_norms=[5e-4], lr=0.01,
              use_bias=False):

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
            adj = [Input(batch_shape=[None, None],
                         dtype=self.floatx, sparse=True, name=f'adj_matrix_{i}') for i in range(self.order+1)]

            index = Input(batch_shape=[None],  dtype=self.intx, name='index')

            h = x
            for hid, activation, dropout, l2_norm in zip(hiddens, activations, dropouts, l2_norms):
                h = ChebyConvolution(hid, order=self.order, use_bias=use_bias, activation=activation,
                                     kernel_regularizer=regularizers.l2(l2_norm))([h, adj])
                h = Dropout(rate=dropout)(h)

            h = ChebyConvolution(self.n_classes, order=self.order, use_bias=use_bias)([h, adj])
            h = Gather()([h, index])

            model = Model(inputs=[x, *adj, index], outputs=h)
            model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                          optimizer=Adam(lr=lr), metrics=['accuracy'])
            self.model = model

    def train_sequence(self, index):
        index = asintarr(index)
        labels = self.labels[index]
        with tf.device(self.device):
            sequence = FullBatchNodeSequence([self.x_norm, *self.adj_norm, index], labels)
        return sequence

    def predict(self, index):
        super().predict(index)
        index = asintarr(index)
        with tf.device(self.device):
            logit = self.model.predict_on_batch([self.x_norm, *self.adj_norm, index])

        if tf.is_tensor(logit):
            logit = logit.numpy()
        return logit
