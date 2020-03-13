import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from graphgallery.nn.layers import ChebyConvolution
from graphgallery.mapper import FullBatchNodeSequence
from graphgallery.utils import chebyshev_polynomials
from .base import SupervisedModel


class ChebyGCN(SupervisedModel):
    
    def __init__(self, adj, features, labels, order=2, normalize_rate=-0.5, 
                 normalize_features=True, device='CPU:0', seed=None):
    
        super().__init__(adj, features, labels, 
                         device=device, seed=seed)
        
        self.order = order
        self.normalize_rate = normalize_rate
        self.normalize_features = normalize_features
        self.preprocess(adj, features)
        
    def preprocess(self, adj, features):
        
        if self.normalize_rate is not None:
            adj = chebyshev_polynomials(adj, rate=self.normalize_rate, order=self.order)
            
        if self.normalize_features:
            features = self._normalize_features(features)

        self.features, self.adj = self._to_tensor([features, adj])  

        
    def build(self, hidden_layers=[32], activations=['relu'], dropout=0.5, learning_rate=0.01, l2_norm=5e-4):
        
        with self.device:
            
            x = Input(batch_shape=[self.n_nodes, self.n_features], dtype=tf.float32, name='features')
            adj = [Input(batch_shape=[self.n_nodes, self.n_nodes], 
                         dtype=tf.float32, sparse=True, name=f'adj_matrix_{i}') for i in range(self.order+1)]
            
            index = Input(batch_shape=[None],  dtype=tf.int32, name='index')

            h = x
            for hid, activation in zip(hidden_layers, activations):
                h = ChebyConvolution(hid, order=self.order, activation=activation, kernel_regularizer=regularizers.l2(l2_norm))([h, adj])
                h = Dropout(rate=dropout)(h)

            h = ChebyConvolution(self.n_classes, order=self.order)([h, adj])
            h = tf.ensure_shape(h, [self.n_nodes, self.n_classes])            
            h = tf.gather(h, index)
            output = Softmax()(h)
            
            model = Model(inputs=[x, *adj, index], outputs=output)
            model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
            self.model = model
            self.built = True
            
    def train_sequence(self, index):
        if self._is_iterable(index):
            return [self.train_sequence(idx) for idx in index]
        else:
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