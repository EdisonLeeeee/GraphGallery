import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from time import perf_counter

from graphgallery.nn.layers import SGConvolution
from graphgallery.mapper import FullBatchNodeSequence
from .base import SupervisedModel

class SGC(SupervisedModel):
    
    def __init__(self, adj, features, labels, order=2, 
                 normalize_rate=-0.5, normalize_features=True, device='CPU:0', seed=None):
    
        super().__init__(adj, features, labels, device=device, seed=seed)
        
        self.order = order
        self.normalize_rate = normalize_rate
        self.normalize_features = normalize_features            
        self.preprocess(adj, features)
        
    def preprocess(self, adj, features):
        
        if self.normalize_rate is not None:
            adj = self._normalize_adj(adj, self.normalize_rate)

        if self.normalize_features:
            features = self._normalize_features(features)

        adj, features = self._to_tensor([adj, features])

        begin_time = perf_counter()

        with self.device:
            features = SGConvolution(order=self.order)([features, adj])

        end_time = perf_counter()
        
        with self.device:
            self.features, self.adj = features, adj
            
        self.precompute_time = end_time - begin_time

        
    def build(self, learning_rate=0.2, l2_norm=5e-5):
        
        with self.device:
            
            x = Input(batch_shape=[None, self.n_features], dtype=tf.float32, name='features')

            output = Dense(self.n_classes, activation='softmax', kernel_regularizer=regularizers.l2(l2_norm))(x)
            
            model = Model(inputs=x, outputs=output)
            model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])

            self.model = model
            self.built = True


    
    def train_sequence(self, index):
        index = self._check_and_convert(index)
        labels = self.labels[index]
        features = tf.gather(self.features, index)
        with self.device:
            sequence = FullBatchNodeSequence(features, labels)
        return sequence
        
    def predict(self, index):
        super().predict(index)
        index = self._check_and_convert(index)
        features = tf.gather(self.features, index)
        with self.device:
            logit = self.model.predict_on_batch(features)
        
        return logit.numpy()