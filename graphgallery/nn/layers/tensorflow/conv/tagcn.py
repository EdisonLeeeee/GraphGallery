from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras.layers import Layer

import tensorflow as tf


class TAGConv(Layer):

    def __init__(self, units,
                 K=3,
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

        super().__init__(**kwargs)
        self.units = units
        self.K = K
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
        self.kernel = self.add_weight(shape=(input_shapes[0][-1] * (self.K + 1), self.units),
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

    def call(self, inputs):

        x, adj = inputs
        xs = [x]
        h = x
        for _ in range(self.K):
            h = tf.sparse.sparse_dense_matmul(adj, h)
            xs.append(h)

        h = tf.concat(xs, axis=-1)
        output = tf.matmul(h, self.kernel)

        if self.use_bias:
            output += self.bias

        return self.activation(output)

    def get_config(self):
        config = {'units': self.units,
                  'K': self.K,
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
        attributes_shape = input_shapes[0]
        output_shape = (attributes_shape[0], self.units)
        return tf.TensorShape(output_shape)  # (num_nodes, output_dim)
