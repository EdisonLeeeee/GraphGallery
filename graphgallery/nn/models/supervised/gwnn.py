import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from graphgallery.nn.layers import WaveletConvolution
from graphgallery.nn.models import SupervisedModel
from graphgallery.sequence import FullBatchNodeSequence
from graphgallery.utils.wavelet_utils import wavelet_basis
from graphgallery.utils.shape_utils import set_equal_in_length
from graphgallery import astensor, asintarr, normalize_x, Bunch


class GWNN(SupervisedModel):
    """
        Implementation of Graph Wavelet Neural Networks (GWNN). 
        `Graph Wavelet Neural Network <https://arxiv.org/abs/1904.07785>`
        Tensorflow 1.x implementation: <https://github.com/Eilene/GWNN>
        Pytorch implementation: <https://github.com/benedekrozemberczki/GraphWaveletNeuralNetwork>

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
                The power (order) of approximated wavelet matrix using Chebyshev polynomial 
                filter. (default :obj: `3`)
            wavelet_s (Float scalar, optional): 
                The wavelet constant corresponding to a heat kernel. 
                (default: :obj:`1.2`) 
            threshold (Float scalar, optional): 
                Used to sparsify the wavelet matrix. (default: :obj:`1e-4`, i.e., 
                values less than `1e-4` will be set to zero to preserve the sparsity 
                of wavelet matrix)       
            wavelet_normalize (Boolean, optional): 
                Whether to use row-normalize for wavelet matrix. (default :bool: `True`)
            norm_x_type (String, optional): 
                How to normalize the node feature matrix. See `graphgallery.normalize_x`
                (default :str: `l1`)
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

    def __init__(self, adj, x, labels, order=3, wavelet_s=1.2,
                 threshold=1e-4, wavelet_normalize=True, norm_x_type='l1',
                 device='CPU:0', seed=None, name=None, **kwargs):

        super().__init__(adj, x, labels, device=device, seed=seed, name=name, **kwargs)

        self.order = order
        self.wavelet_s = wavelet_s
        self.threshold = threshold
        self.wavelet_normalize = wavelet_normalize
        self.norm_x_type = norm_x_type
        self.preprocess(adj, x)

    def preprocess(self, adj, x):
        super().preprocess(adj, x)
        # check the input adj and x, and convert them into proper data types
        adj, x = self._check_inputs(adj, x)

        if self.norm_x_type:
            x = normalize_x(x, norm=self.norm_x_type)

        wavelet, inverse_wavelet = wavelet_basis(adj, wavelet_s=self.wavelet_s,
                                                 order=self.order, threshold=self.threshold,
                                                 wavelet_normalize=self.wavelet_normalize)
        with tf.device(self.device):
            self.x_norm, self.adj_norm = astensor([x, [wavelet, inverse_wavelet]])

    def build(self, hiddens=[16], activations=['relu'], dropouts=[0.5], l2_norms=[5e-4], lr=0.01,
              ensure_shape=True):

        local_paras = locals()
        local_paras.pop('self')
        paras = Bunch(**local_paras)
        hiddens, activations, dropouts, l2_norms = set_equal_in_length(hiddens, activations, dropouts, l2_norms)
        paras.update(Bunch(hiddens=hiddens, activations=activations, dropouts=dropouts, l2_norms=l2_norms))
        # update all parameters
        self.paras.update(paras)
        self.model_paras.update(paras)

        with tf.device(self.device):

            x = Input(batch_shape=[self.n_nodes, self.n_features], dtype=self.floatx, name='features')
            wavelet = Input(batch_shape=[self.n_nodes, self.n_nodes], dtype=self.floatx, sparse=True, name='wavelet')
            inverse_wavelet = Input(batch_shape=[self.n_nodes, self.n_nodes], dtype=self.floatx, sparse=True,
                                    name='inverse_wavelet')
            index = Input(batch_shape=[None],  dtype=self.intx, name='index')

            h = x
            for hid, activation, dropout, l2_norm in zip(hiddens, activations, dropouts, l2_norms):
                h = WaveletConvolution(hid, activation=activation,
                                       kernel_regularizer=regularizers.l2(l2_norm))([h, wavelet, inverse_wavelet])
                h = Dropout(rate=dropout)(h)

            h = WaveletConvolution(self.n_classes)([h, wavelet, inverse_wavelet])
            # To aviod the UserWarning of `tf.gather`, but it causes the shape
            # of the input data to remain the same
            if ensure_shape:
                h = tf.ensure_shape(h, [self.n_nodes, self.n_classes])
            h = tf.gather(h, index)
            output = Softmax()(h)

            model = Model(inputs=[x, wavelet, inverse_wavelet, index], outputs=output)
            model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])

            self.set_model(model)

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
            index = astensor(index)
            logit = self.model.predict_on_batch([self.x_norm, *self.adj_norm, index])

        if tf.is_tensor(logit):
            logit = logit.numpy()
        return logit
