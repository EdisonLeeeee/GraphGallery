import tensorflow as tf
from tensorflow.keras.layers import Layer


class SparseConversion(Layer):
    def __init__(self, num_nodes=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_nodes = num_nodes
        self.trainable = False

    def call(self, inputs):
        indices, values = inputs
        num_nodes = tf.reduce_max(indices) + 1

        self.num_nodes = num_nodes

        if indices.dtype != tf.int64:
            indices = tf.cast(indices, tf.int64)

        # Build sparse tensor for the matrix
        output = tf.sparse.SparseTensor(
            indices=indices, values=values, dense_shape=(num_nodes, num_nodes)
        )
        return output

    def compute_output_shape(self, input_shapes):
        return tf.TensorShape([self.num_nodes, self.num_nodes])

    def get_config(self):
        base_config = super().get_config()
        return base_config


class Scale(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainable = False

    def call(self, input):
        output = (input - tf.reduce_mean(input, axis=0, keepdims=True)
                  ) / tf.keras.backend.std(input, axis=0, keepdims=True)
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
        self.trainable = False

    def call(self, inputs):
        mean, var = inputs
        sample = tf.random.normal(tf.shape(var), 0, 1, dtype=mean.dtype)
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
        self.trainable = False

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
        output_shape = params_shape[:axis] + \
            indices_shape + params_shape[axis + 1:]
        return tf.TensorShape(output_shape)


class Mask(Layer):
    def __init__(self, axis=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.axis = axis
        self.trainable = False

    def call(self, inputs):
        tensor, mask = inputs
        output = tf.boolean_mask(tensor, mask, axis=self.axis)
        return output

    def get_config(self):
        base_config = super().get_config()
        return base_config


class Laplacian(Layer):
    def __init__(self, rate=-0.5, fill_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not rate:
            raise ValueError(
                f"`rate` must be an integer scalr larger than zero, but got {rate}.")
        self.rate = rate
        self.fill_weight = fill_weight
        self.trainable = False

    def call(self, adj):
        if self.fill_weight:
            adj = adj + self.fill_weight * \
                tf.eye(tf.shape(adj)[0], dtype=adj.dtype)
        d = tf.reduce_sum(adj, axis=1)
        d_power = tf.pow(d, self.rate)
        d_power_mat = tf.linalg.diag(d_power)
        return d_power_mat @ adj @ d_power_mat

    def get_config(self):
        base_config = super().get_config()
        return base_config

    def compute_output_shape(self, input_shapes):
        return tf.TensorShape(input_shapes)
