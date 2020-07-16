from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras.layers import Layer, Dropout, LeakyReLU

import tensorflow as tf


class DenseGraphConv(Layer):
    """
        Basic graph convolution layer as in: 
        `Semi-Supervised Classification with Graph Convolutional Networks` (https://arxiv.org/abs/1609.02907)
        Tensorflow 1.x implementation: https://github.com/tkipf/gcn
        Pytorch implementation: https://github.com/tkipf/pygcn

        `DenseGraphConv` implements the `Dense` operation:
        `output = activation(adj @ x @ kernel + bias)`
        where `x` is the feature matrix, `adj` is the adjacency matrix with dense form,
        `activation` is the element-wise activation function
        passed as the `activation` argument, `kernel` is a weights matrix
        created by the layer, and `bias` is a bias vector created by the layer
        (only applicable if `use_bias` is `True`).


        Arguments:
          units: Positive integer, dimensionality of the output space.
          activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
          use_bias: Boolean, whether the layer uses a bias vector.
          kernel_initializer: Initializer for the `kernel` weights matrix.
          bias_initializer: Initializer for the bias vector.
          kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
          bias_regularizer: Regularizer function applied to the bias vector.
          activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
          kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
          bias_constraint: Constraint function applied to the bias vector.

        Input shape:
          tuple/list with two 2-D tensor: Tensor `x` and SparseTensor `adj`: `[(n_nodes, n_features), (n_nodes, n_nodes)]`.
          The former one is the feature matrix (Tensor) and the last is adjacency matrix (Tensor).

        Output shape:
          2-D tensor with shape: `(n_nodes, units)`.       
    """

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

    def call(self, inputs):

        x, adj = inputs
        h = x @ self.kernel
        output = adj @ h

        if self.use_bias:
            output += self.bias

        return self.activation(output)

    def get_config(self):
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
        features_shape = input_shapes[0]
        output_shape = (features_shape[0], self.units)
        return tf.TensorShape(output_shape)  # (n_nodes, output_dim)
