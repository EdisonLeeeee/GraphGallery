from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras.layers import Layer, Dropout, LeakyReLU

import tensorflow as tf


class ChebyConvolution(Layer):
    """
        Basic Chebyshev graph convolution layer as in: 
        [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375)
        Tensorflow 1.x implementation: https://github.com/mdeff/cnn_graph, https://github.com/tkipf/gcn
        Keras implementation: https://github.com/aclyde11/ChebyGCN

        `ChebyConvolution` implements the operation:
        `output = activation(adj_0 @ x @ kernel + bias) + activation(adj_1 @ x @ kernel + bias) + ...`
        where `x` is the feature matrix, `adj_i` is the i-th adjacency matrix, 0<=i<=`order`,
        `activation` is the element-wise activation function
        passed as the `activation` argument, `kernel` is a weights matrix
        created by the layer, and `bias` is a bias vector created by the layer
        (only applicable if `use_bias` is `True`).


        Arguments:
          units: Positive integer, dimensionality of the output space.
          order: Positive integer, the order of `Chebyshev polynomials`.
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
          tuple/list with `order + 2` 2-D tensor: Tensor `x` and `order + 1` SparseTensor `adj`: 
          `[(n_nodes, n_features), (n_nodes, n_nodes), (n_nodes, n_nodes), ...]`.
          The former one is the feature matrix (Tensor) and the last is adjacency matrix (SparseTensor).

        Output shape:
          2-D tensor with shape: `(n_nodes, units)`.  
    """

    def __init__(self, units,
                 order,
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
        self.order = order
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
        self.kernel, self.bias = [], []
        input_dim = input_shapes[0][-1]
        for i in range(self.order+1):
            kernel = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.kernel_initializer,
                                     name=f'kernel_{i}',
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)
            if self.use_bias:
                bias = self.add_weight(shape=(self.units,),
                                       initializer=self.bias_initializer,
                                       name=f'bias_{i}',
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint)
            else:
                bias = None
            self.kernel.append(kernel)
            self.bias.append(bias)

        super().build(input_shapes)

    def call(self, inputs):

        x, adjs = inputs
        supports = []
        for adj, kernel, bias in zip(adjs, self.kernel, self.bias):
            support = x @ kernel
            support = tf.sparse.sparse_dense_matmul(adj, support)
            if self.use_bias:
                support += bias
            supports.append(support)

        output = tf.add_n(supports)

        return self.activation(output)

    def get_config(self):
        config = {'units': self.units,
                  'order': self.order,
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
