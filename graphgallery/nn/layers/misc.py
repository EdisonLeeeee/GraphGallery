import tensorflow as tf
from tensorflow.keras.layers import Layer
from graphgallery import config


class SparseConversion(Layer):
    def __init__(self, n_nodes=None):
        super().__init__()
        self.n_nodes = n_nodes

    def call(self, inputs):
        indices, values = inputs
        n_nodes = tf.reduce_max(indices) + 1

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


    def get_config(self):
        base_config = super().get_config()
        return base_config

class Scale(Layer):
    def call(self, inputs):
        output = (inputs - tf.reduce_mean(inputs, axis=0, keepdims=True)) / tf.keras.backend.std(inputs, axis=0, keepdims=True)
        return output

    def get_config(self):
        base_config = super().get_config()
        return base_config

    def compute_output_shape(self, input_shapes):
        return tf.TensorShape(input_shapes)


class Sample(Layer):
    def __init__(self, seed=None, *args, **kwargs):
        if seed:
            tf.random.set_seed(seed)
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        mean, var = inputs
        sample = tf.random.normal(tf.shape(var), 0, 1, dtype=config.floatx())
        output = mean + tf.math.sqrt(var + 1e-8) * sample
        return output

    def get_config(self):
        base_config = super().get_config()
        return base_config

    def compute_output_shape(self, input_shapes):
        return tf.TensorShape(input_shapes[0])


class Gather(Layer):
    def __init__(self, axis=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.axis = axis

    def call(self, inputs):
        params, indices = inputs
        output = tf.gather(params, indices, axis=self.axis)
        return output

    def get_config(self):
        base_config = super().get_config()
        return base_config

    def compute_output_shape(self, input_shapes):
        axis = self.axis
        params_shape, indices_shape = input_shapes
        output_shape = params_shape[:axis] + indices_shape + params_shape[axis + 1:]
        return tf.TensorShape(output_shape)
