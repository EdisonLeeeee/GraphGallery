import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from graphgallery.nn.layers import WaveletConvolution
from graphgallery.mapper import FullBatchNodeSequence
from graphgallery.utils import wavelet_basis
from .base import SupervisedModel

class GWNN(SupervisedModel):
    """
        Implementation of Graph Wavelet Neural Networks (GWNN). 
        [Graph Wavelet Neural Network](https://arxiv.org/abs/1904.07785)
        Tensorflow 1.x implementation: https://github.com/Eilene/GWNN
        Pytorch implementation: https://github.com/benedekrozemberczki/GraphWaveletNeuralNetwork

        Arguments:
        ----------
            adj: `scipy.sparse.csr_matrix` (or `csc_matrix`) with shape (N, N)
                The input `symmetric` adjacency matrix, where `N` is the number of nodes 
                in graph.
            features: `np.array` with shape (N, F)
                The input node feature matrix, where `F` is the dimension of node features.
            labels: `np.array` with shape (N,)
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
                Whether to use row-normalize for wavelet matrix. (default :obj: `True`)
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
    
    def __init__(self, adj, features, labels, order=3, wavelet_s=1.2, 
                 threshold=1e-4, wavelet_normalize=True, normalize_features=True, 
                 device='CPU:0', seed=None, **kwargs):
    
        super().__init__(adj, features, labels, device=device, seed=seed, **kwargs)
        
        self.order = order
        self.wavelet_s = wavelet_s
        self.threshold = threshold
        self.wavelet_normalize = wavelet_normalize
        self.normalize_features = normalize_features   
        self.preprocess(adj, features)
        
    def preprocess(self, adj, features):
        
        if self.normalize_features:
            features = self._normalize_features(features)
            
        wavelet, inverse_wavelet = wavelet_basis(adj, wavelet_s=self.wavelet_s, 
                                                 order=self.order, threshold=self.threshold, 
                                                 wavelet_normalize=self.wavelet_normalize)
        with self.device:
            self.features, self.adj = self._to_tensor([features, [wavelet, inverse_wavelet]])
        
    def build(self, hidden_layers=[16], activations=['relu'], dropout=0.5, learning_rate=0.01, l2_norm=5e-4):
        
        with self.device:
            
            x = Input(batch_shape=[self.n_nodes, self.n_features], dtype=tf.float32, name='features')
            wavelet = Input(batch_shape=[self.n_nodes, self.n_nodes], dtype=tf.float32, sparse=True, name='wavelet')           
            inverse_wavelet = Input(batch_shape=[self.n_nodes, self.n_nodes], dtype=tf.float32, sparse=True,
                                    name='inverse_wavelet')
            index = Input(batch_shape=[None],  dtype=tf.int32, name='index')

            h = x

            for hid, activation in zip(hidden_layers, activations):
                h = WaveletConvolution(hid, activation=activation, 
                                       kernel_regularizer=regularizers.l2(l2_norm))([h, wavelet, inverse_wavelet])
                h = Dropout(rate=dropout)(h)

            h = WaveletConvolution(self.n_classes)([h, wavelet, inverse_wavelet])
            h = tf.ensure_shape(h, [self.n_nodes, self.n_classes])            
            h = tf.gather(h, index)
            output = Softmax()(h)

            model = Model(inputs=[x, wavelet, inverse_wavelet, index], outputs=output)
            model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])

            self.model = model
            self.built = True
            
    def train_sequence(self, index):
        index = self._check_and_convert(index)
        labels = self.labels[index]  

        with self.device:
            sequence = FullBatchNodeSequence([self.features, *self.adj, index], labels)
        return sequence
        
    
    def predict(self, index):
        super().predict(index)
        index = self._check_and_convert(index)
        with self.device:
            index = self._to_tensor(index)
            logit = self.model.predict_on_batch([self.features, *self.adj, index])

        return logit.numpy()    