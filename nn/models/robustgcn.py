import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from graphgallery.nn.layers import GaussionConvolution_F, GaussionConvolution_D
from graphgallery.mapper import FullBatchNodeSequence
from .base import SupervisedModel


class RobustGCN(SupervisedModel):
    
    def __init__(self, adj, features, labels, normalize_rate=[-0.5, -1], normalize_features=True, device='CPU:0', seed=None):
    
        super().__init__(adj, features, labels, device=device, seed=seed)
        
        self.normalize_rate = normalize_rate
        self.normalize_features = normalize_features            
        self.preprocess(adj, features)
            
    def preprocess(self, adj, features):
        
        if self.normalize_rate is not None:
            adj = self._normalize_adj([adj, adj], self.normalize_rate)    # [adj_1, adj_2]    
            
        if self.normalize_features:
            features = self._normalize_features(features)
            
        with self.device:
            self.features, self.adj = self._to_tensor([features, adj])
        
    def build(self, hidden_layers=[64], activations=['relu'], dropout=0.5, learning_rate=0.01, l2_norm=5e-4, para_kl=5e-4):
        
        x = Input(batch_shape=[self.n_nodes, self.n_features], dtype=tf.float32, name='features')
        adj = [Input(batch_shape=[self.n_nodes, self.n_nodes], dtype=tf.float32, sparse=True, name='adj_matrix_1'),
               Input(batch_shape=[self.n_nodes, self.n_nodes], dtype=tf.float32, sparse=True, name='adj_matrix_2')]
        index = Input(batch_shape=[None],  dtype=tf.int32, name='index')

        h, KL_divergence = GaussionConvolution_F(hidden_layers[0], 
                                                      activation=activations[0], 
                                                      kernel_regularizer=regularizers.l2(l2_norm))([x, *adj])
        h = Dropout(rate=dropout)(h)
        
        # additional layers (usually unnecessay)
        for hid, activation in zip(hidden_layers[1:], activations[1:]):
            h = GaussionConvolution_D(hid, activation=activation)([h, *adj])
            h = Dropout(rate=dropout)(h)
            
        h = GaussionConvolution_D(self.n_classes)([h, *adj])
        h = tf.ensure_shape(h, [self.n_nodes, self.n_classes])            
        h = tf.gather(h, index)
        output = Softmax()(h)

        model = Model(inputs=[x, *adj, index], outputs=output)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
        model.add_loss(para_kl * tf.reduce_sum(KL_divergence))
        
        self.model = model
        self.built = True
        
    def train_sequence(self, index):
        index = self._check_and_convert(index)
        labels = self.labels[index]      
        with self.device:
            sequence = FullBatchNodeSequence([self.features, *self.adj, index], labels)
        return sequence

    def predict(self, index):
        if not self.built:
            raise RuntimeError('You must compile your model before training/testing/predicting. Use `model.build()`.')

        if self.do_before_predict is not None:
            self.do_before_predict(idx, **kwargs)

        index = self._check_and_convert(index)

        with self.device:
            index = self._to_tensor(index)
            logit = self.model.predict_on_batch([self.features, *self.adj, index])

        return logit.numpy()    