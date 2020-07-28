from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras.layers import Layer, Dropout, LeakyReLU

import tensorflow as tf
import numpy as np


class WaveletConvolution(Layer):
    """
        Basic graph convolution layer as in: 
        [Graph Wavelet Neural Network](https://arxiv.org/abs/1904.07785)
        Tensorflow 1.x implementation: https://github.com/Eilene/GWNN
        Pytorch implementation: https://github.com/benedekrozemberczki/GraphWaveletNeuralNetwork

        `WaveletConvolution` implements the operation:
        `output = activation(wavelet @ filter @ inverse_wavelet @ x @ kernel + bias)`
        where `x` is the feature matrix, `wavelet` is the wavelet matrix,
        `inverse_wavelet` is the inversed wavelet matrix, filter is the trainable diagnal matrix.
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
          tuple/list with three 2-D tensor: Tensor `x`, SparseTensor `wavelet` and inverse_wavelet: `[(n_nodes, n_features), (n_nodes, n_nodes), (n_nodes, n_nodes)]`.
          The former one is the feature matrix (Tensor) and the others is wavelet matrix (SparseTensor) and inversed wavelet matrix (SparseTensor).

        Output shape:
          2-D tensor with shape: `(n_nodes, units)`.       
    """

    def __init__(self, units,
                 use_bias=False,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 filter_initializer='ones',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 filter_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 filter_constraint=None,
                 **kwargs):

        super().__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias

        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.filter_initializer = initializers.get(filter_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.filter_regularizer = regularizers.get(filter_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.filter_constraint = constraints.get(filter_constraint)

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

        n_nodes = input_shapes[1][0]
        self.filter = self.add_weight(shape=(n_nodes,),
                                      initializer=self.filter_initializer,
                                      name='filter',
                                      regularizer=self.filter_regularizer,
                                      constraint=self.filter_constraint)

        self.indices = np.stack([np.arange(n_nodes)]*2, axis=1)

        super().build(input_shapes)

    def call(self, inputs):

        filter_ = tf.sparse.SparseTensor(indices=self.indices,
                                         values=self.filter,
                                         dense_shape=self.indices[-1] + 1)

        x, wavelet, inverse_wavelet = inputs
        h = x @ self.kernel
        h = tf.sparse.sparse_dense_matmul(inverse_wavelet, h)
        h = tf.sparse.sparse_dense_matmul(filter_, h)
        output = tf.sparse.sparse_dense_matmul(wavelet, h)

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
                  'filter_initializer': initializers.serialize(
                      self.filter_initializer),
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(
                      self.bias_regularizer),
                  'filter_regularizer': regularizers.serialize(
                      self.filter_regularizer),
                  'activity_regularizer': regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'bias_constraint': constraints.serialize(
                      self.bias_constraint),
                  'filter_constraint': constraints.serialize(
                      self.filter_constraint)
                  }

        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shapes):
        features_shape = input_shapes[0]
        output_shape = (features_shape[0], self.units)
        return tf.TensorShape(output_shape)  # (batch_size, output_dim)
