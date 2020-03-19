import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class Top_k_features(Layer):
    def __init__(self, k, **kwargs):
        
        super().__init__(**kwargs)
        self.k = k
        
    def call(self, inputs):
        
        x, adj = inputs
        if K.is_sparse(adj):
            adj = tf.sparse.to_dense(adj)
        adj = tf.expand_dims(adj, axis=1)
        x = tf.expand_dims(x, axis=-1)
        h = adj * x
        h = tf.transpose(h, perm=(2, 1, 0))
        h = tf.math.top_k(h, k=self.k, sorted=True).values
        h = tf.concat([x, h], axis=-1)
        h = tf.transpose(h, perm=(0, 2, 1))
        return h  
            
    def get_config(self):
        config = {'k': self.k}

        base_config = super().get_config()
        return {**base_config, **config}
    
    
    def compute_output_shape(self, input_shapes):
        features_shape = input_shapes[0]
        output_shape = (features_shape[0], self.k+1, features_shape[1])
        return output_shape  
    
                             
