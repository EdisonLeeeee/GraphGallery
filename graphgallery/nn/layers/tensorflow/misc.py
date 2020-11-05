import tensorflow as tf
from tensorflow.keras.layers import Layer


class SparseConversion(Layer):
    def __init__(self, n_nodes=None, *args, **kwargs):
        """
        Initialize the graph.

        Args:
            self: (todo): write your description
            n_nodes: (int): write your description
        """
        super().__init__(*args, **kwargs)
        self.n_nodes = n_nodes
        self.trainable = False

    def call(self, inputs):
        """
        See tf.

        Args:
            self: (todo): write your description
            inputs: (dict): write your description
        """
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
        """
        Compute the output shape.

        Args:
            self: (todo): write your description
            input_shapes: (list): write your description
        """
        return tf.TensorShape([self.n_nodes, self.n_nodes])

    def get_config(self):
        """
        Returns the base collector settings

        Args:
            self: (str): write your description
        """
        base_config = super().get_config()
        return base_config


class Scale(Layer):
    def __init__(self, *args, **kwargs):
        """
        Initialize the class.

        Args:
            self: (todo): write your description
        """
        super().__init__(*args, **kwargs)
        self.trainable = False

    def call(self, input):
        """
        Compute the kl divergence.

        Args:
            self: (todo): write your description
            input: (array): write your description
        """
        output = (input - tf.reduce_mean(input, axis=0, keepdims=True)
                  ) / tf.keras.backend.std(input, axis=0, keepdims=True)
        return output

    def get_config(self):
        """
        Returns the base collector settings

        Args:
            self: (str): write your description
        """
        base_config = super().get_config()
        return base_config

    def compute_output_shape(self, input_shapes):
        """
        Compute the output shape.

        Args:
            self: (todo): write your description
            input_shapes: (list): write your description
        """
        return tf.TensorShape(input_shapes)


class Sample(Layer):
    def __init__(self, seed=None, *args, **kwargs):
        """
        Initialize a random seed.

        Args:
            self: (todo): write your description
            seed: (int): write your description
        """
        if seed:
            tf.random.set_seed(seed)
        super().__init__(*args, **kwargs)
        self.trainable = False

    def call(self, inputs):
        """
        Return a new variable.

        Args:
            self: (todo): write your description
            inputs: (dict): write your description
        """
        mean, var = inputs
        sample = tf.random.normal(tf.shape(var), 0, 1, dtype=mean.dtype)
        output = mean + tf.math.sqrt(var + 1e-8) * sample
        return output

    def get_config(self):
        """
        Returns the base collector settings

        Args:
            self: (str): write your description
        """
        base_config = super().get_config()
        return base_config

    def compute_output_shape(self, input_shapes):
        """
        Computes the shape.

        Args:
            self: (todo): write your description
            input_shapes: (list): write your description
        """
        return tf.TensorShape(input_shapes[0])


class Gather(Layer):
    def __init__(self, axis=0, *args, **kwargs):
        """
        Initialize kwargs.

        Args:
            self: (todo): write your description
            axis: (int): write your description
        """
        super().__init__(*args, **kwargs)
        self.axis = axis
        self.trainable = False

    def call(self, inputs):
        """
        Evaluate the model.

        Args:
            self: (todo): write your description
            inputs: (dict): write your description
        """
        params, indices = inputs
        output = tf.gather(params, indices, axis=self.axis)
        return output

    def get_config(self):
        """
        Returns the base collector settings

        Args:
            self: (str): write your description
        """
        base_config = super().get_config()
        return base_config

    def compute_output_shape(self, input_shapes):
        """
        Compute the shape.

        Args:
            self: (todo): write your description
            input_shapes: (list): write your description
        """
        axis = self.axis
        params_shape, indices_shape = input_shapes
        output_shape = params_shape[:axis] + \
            indices_shape + params_shape[axis + 1:]
        return tf.TensorShape(output_shape)


class Mask(Layer):
    def __init__(self, axis=None, *args, **kwargs):
        """
        Initialize an axis.

        Args:
            self: (todo): write your description
            axis: (int): write your description
        """
        super().__init__(*args, **kwargs)
        self.axis = axis
        self.trainable = False

    def call(self, inputs):
        """
        Applies a tensor.

        Args:
            self: (todo): write your description
            inputs: (dict): write your description
        """
        tensor, mask = inputs
        output = tf.boolean_mask(tensor, mask, axis=self.axis)
        return output

    def get_config(self):
        """
        Returns the base collector settings

        Args:
            self: (str): write your description
        """
        base_config = super().get_config()
        return base_config


class Laplacian(Layer):
    def __init__(self, rate=-0.5, fill_weight=1.0, *args, **kwargs):
        """
        Initialize the weights.

        Args:
            self: (todo): write your description
            rate: (todo): write your description
            fill_weight: (int): write your description
        """
        super().__init__(*args, **kwargs)
        if not rate:
            raise ValueError(
                f"`rate` must be an integer scalr larger than zero, but got {rate}.")
        self.rate = rate
        self.fill_weight = fill_weight
        self.trainable = False

    def call(self, adj):
        """
        Implement the adjacency matrix.

        Args:
            self: (todo): write your description
            adj: (array): write your description
        """
        if self.fill_weight:
            adj = adj + self.fill_weight * \
                tf.eye(tf.shape(adj)[0], dtype=adj.dtype)
        d = tf.reduce_sum(adj, axis=1)
        d_power = tf.pow(d, self.rate)
        d_power_mat = tf.linalg.diag(d_power)
        return d_power_mat @ adj @ d_power_mat

    def get_config(self):
        """
        Returns the base collector settings

        Args:
            self: (str): write your description
        """
        base_config = super().get_config()
        return base_config

    def compute_output_shape(self, input_shapes):
        """
        Compute the output shape.

        Args:
            self: (todo): write your description
            input_shapes: (list): write your description
        """
        return tf.TensorShape(input_shapes)
