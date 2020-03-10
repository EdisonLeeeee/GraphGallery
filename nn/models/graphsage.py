import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from graphgallery.nn.layers import SAGEConvolution
from graphgallery.mapper import SAGEMiniBatchSequence
from graphgallery.utils import construct_adj
from .base import SupervisedModel


class GraphSAGE(SupervisedModel):
    
    
    def __init__(self, adj, features, labels, normalize_features=False, n_samples=[10, 5], device='CPU:0', seed=None):
    
        super().__init__(adj, features, labels, device=device, seed=seed)
        
        if normalize_features:
            features = self._normalize_features(features)
            
        adj = construct_adj(adj, max_degree=max(n_samples)) # Dense matrix, shape [n_nodes, max_degree]
        # pad with dummy zero vector
        features = np.vstack([features, np.zeros(self.n_features, dtype=np.float32)])
        features = self._to_tensor(features)
        self.features, self.adj = features, adj
        self.n_samples = n_samples

    def build(self, hidden_layers=[16], activations=['elu'], dropout=0.5, learning_rate=0.01, l2_norm=5e-4, 
              output_normalize=False, agg_method='mean'):
        
        with self.device:
            
            x = Input(batch_shape=[None, self.n_features], dtype=tf.float32, name='features')
            nodes = Input(batch_shape=[None], dtype=tf.int32, name='nodes')
            neighbors = [Input(batch_shape=[None], dtype=tf.int32, name=f'neighbors_{hop}') 
                         for hop, n_sample in enumerate(self.n_samples)]
            
            aggrators = []
            for i, (hid, activation) in enumerate(zip(hidden_layers, activations)):
                aggrators.append(SAGEConvolution(hid, concat=True, agg_method=agg_method, 
                                          activation=activation, 
                                          kernel_regularizer=regularizers.l2(l2_norm)))
            aggrators.append(SAGEConvolution(self.n_classes, agg_method=agg_method))
                
            h = [tf.nn.embedding_lookup(x, node) for node in [nodes, *neighbors]]
            for agg_i, aggrator  in enumerate(aggrators):
                feature_shape = h[0].shape[-1]
                for hop in range(len(self.n_samples)-agg_i):
                    neighbor_shape = [-1, self.n_samples[hop], feature_shape]
                    h[hop] = Dropout(rate=dropout)(aggrator([h[hop], tf.reshape(h[hop+1], neighbor_shape)]))
                h.pop()
                
            h = h[0]
            if output_normalize:
                h = tf.nn.l2_normalize(h, axis=1)
            output = Softmax()(h)
                
            model = Model(inputs=[x, nodes, *neighbors], outputs=output)
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
                sequence = SAGEMiniBatchSequence([self.features, index], labels, self.adj, n_samples=self.n_samples)
            return sequence
        
            
    def predict(self, index):
        
        if not self.built:
            raise RuntimeError('You must compile your model before training/testing/predicting. Use `model.build()`.')
            
        index = self._check_and_convert(index)
        logit = []
        with self.device:
            data = SAGEMiniBatchSequence([self.features, index], adj=self.adj, n_samples=self.n_samples)
            for inputs, labels in data:
                output = self.model.predict_on_batch(inputs)
                logit.append(output.numpy())
        logit = np.concatenate(logit, axis=0)
        return logit