import tensorflow as tf
from tensorflow.keras.layers import Layer


class SqueezedSparseConversion(Layer):
    def call(self, inputs):
        indices, values = inputs
        n_nodes = tf.reduce_max(indices) + 1
        
        if indices.dtype != tf.int64:
            indices = tf.cast(indices, tf.int64)

        # Build sparse tensor for the matrix
        output = tf.sparse.SparseTensor(
            indices=indices, values=values, dense_shape=(n_nodes, n_nodes)
        )
        return output