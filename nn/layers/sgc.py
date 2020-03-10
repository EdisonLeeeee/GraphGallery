import tensorflow as tf
from tensorflow.keras.layers import Layer


class SGConvolution(Layer):
    
    def __init__(self, order=2, **kwargs):
        super().__init__(**kwargs)
        self.order = order
        

    def call(self, inputs):
        features, adj = inputs
        
        for _ in range(self.order):
            features = tf.sparse.sparse_dense_matmul(adj, features)

        return features
    
    
    def get_config(self):
        config = {'order': self.order}
        
        base_config = super().get_config()
        return {**base_config, **config}
    
    
    def compute_output_shape(self, input_shapes):
        features_shape = input_shapes[0]
        return features_shape
