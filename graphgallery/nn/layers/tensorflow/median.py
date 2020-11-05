from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras.layers import Layer

import tensorflow as tf
try:
    import tensorflow_probability as tfp
except ImportError:
    tfp = None


class MedianConvolution(Layer):

    def __init__(self, units,
                 use_bias=False,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            units: (todo): write your description
            use_bias: (bool): write your description
            activation: (str): write your description
            kernel_initializer: (int): write your description
            bias_initializer: (int): write your description
            kernel_regularizer: (dict): write your description
            bias_regularizer: (dict): write your description
            activity_regularizer: (bool): write your description
            kernel_constraint: (todo): write your description
            bias_constraint: (str): write your description
        """

        super().__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias

        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shapes):
        """
        Connects the module into the graph.

        Args:
            self: (todo): write your description
            input_shapes: (list): write your description
        """
        self.kernel = self.add_weight(shape=(input_shapes[0][-1], self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        super().build(input_shapes)

    @tf.function
    def call(self, inputs):
        """
        Call tf.

        Args:
            self: (todo): write your description
            inputs: (dict): write your description
        """

        x, neighbors = inputs
        h = x @ self.kernel

        aggregation = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        for neighbor in neighbors:
            msg = tf.gather(h, neighbor)
            agg = tfp.stats.percentile(msg, q=50., axis=0, interpolation='midpoint')
            aggregation = aggregation.write(aggregation.size(), agg)

        output = aggregation.stack()
        if self.use_bias:
            output += self.bias
        return self.activation(output)

    def get_config(self):
        """
        Get the configurations.

        Args:
            self: (str): write your description
        """
        config = {'units': self.units,
                  'use_bias': self.use_bias,
                  'activation': activations.serialize(self.activation),
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'bias_initializer': initializers.serialize(
                      self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(
                      self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint)
                  }

        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shapes):
        """
        Compute the output shape.

        Args:
            self: (todo): write your description
            input_shapes: (list): write your description
        """
        attributes_shape = input_shapes[0]
        output_shape = (attributes_shape[0], self.units)
        return tf.TensorShape(output_shape)  # (n_nodes, output_dim)
