import tensorflow as tf
from tensorflow.keras.layers import Layer
from graphgallery import config


class SqueezedSparseConversion(Layer):
    def __init__(self, n_nodes=None):
        super().__init__()
        self.n_nodes = n_nodes

    def call(self, inputs):
        indices, values = inputs
        n_nodes = self.n_nodes or tf.reduce_max(indices) + 1
        self.n_nodes = n_nodes

        if indices.dtype != tf.int64:
            indices = tf.cast(indices, tf.int64)

        # Build sparse tensor for the matrix
        output = tf.sparse.SparseTensor(
            indices=indices, values=values, dense_shape=(n_nodes, n_nodes)
        )
        return output

    def compute_output_shape(self, input_shapes):
        return tf.TensorShape([self.n_nodes, self.n_nodes])


class Scale(Layer):
    def call(self, inputs):
        output = (inputs - tf.reduce_mean(inputs, axis=0, keepdims=True)) / tf.keras.backend.std(inputs, axis=0, keepdims=True)
        return output

    def compute_output_shape(self, input_shapes):
        return tf.TensorShape(input_shapes)
    
class Sample(Layer):
    def __init__(self, seed=None, *args, **kwargs):
        if seed:
            tf.random.set_seed(seed)
        super().__init__(*args, **kwargs)
        
    def call(self, mean, var):
        sample = tf.random.normal(tf.shape(var), 0, 1, dtype=config.floatx())
        output = mean + tf.math.sqrt(var + 1e-8) * sample
        return output 
    
    def compute_output_shape(self, input_shapes):
        return tf.TensorShape(input_shapes[0])
    
