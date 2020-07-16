
from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras.layers import Layer, Dropout, LeakyReLU

import tensorflow as tf

from graphgallery import config


class GaussionConvolution_F(Layer):
    """
        Robust graph convolution layer as in: 
        [Robust Graph Convolutional Networks Against Adversarial Attacks](https://dl.acm.org/doi/10.1145/3292500.3330851)
        Tensorflow 1.x implementation: https://github.com/thumanlab/nrlweb/blob/master/static/assets/download/RGCN.zip

        `GaussionConvolution_F` implements the GaussionConvolution operation
           where the inputs is node feature matrix and two adjacency matrices,
           the output is concatenated `mean vector` and `variance vector`.

        Arguments:
          units: Positive integer, dimensionality of the output space.
          gamma: float scalar, decide the attention weights for mean and variance. 
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
          tuple/list with three 2-D tensor: Tensor `x` and SparseTensor `adj_0, adj_1`: 
            `[(n_nodes, n_features), (n_nodes, n_nodes), (n_nodes, n_nodes)]`.
          The former one is the feature matrix (Tensor) and the others are adjacency matrix (SparseTensor) with different normalize rate (-0.5, -1.0).

        Output shape:
          ((n_nodes, units), (1,)), the first one is 2-D tensor with shape: `(n_nodes, units)` which is concatenated by 
            `mean vactor (n_nodes, units//2)` and `variance vector (n_nodes, units//2)`.  
          The last one is a scalar `KL_divergence` of output distribution and Gaussian distribution.
    """

    def __init__(self, units,
                 gamma=1.,
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
        self.dim = units // 2
        self.gamma = gamma
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
        self.kernel = self.add_weight(shape=(input_shapes[0][1], self.units),
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
        x, *adj = inputs
#         assert len(adj) == 2

        h = x @ self.kernel

        if self.use_bias:
            h += self.bias

        mean = activations.elu(tf.slice(h, [0, 0], [-1, self.dim]))
        var = activations.relu(tf.slice(h, [0, self.dim], [-1, self.dim]))

        KL_divergence = 0.5 * tf.reduce_mean(tf.math.square(mean) + var - tf.math.log(1e-8 + var) - 1, axis=1)
        KL_divergence = tf.reduce_sum(KL_divergence)

        attention = tf.exp(-self.gamma*var)
        mean = tf.sparse.sparse_dense_matmul(adj[0], mean * attention)
        var = tf.sparse.sparse_dense_matmul(adj[1], var * attention * attention)

        output = tf.concat([mean, var], axis=1)

        return self.activation(output), KL_divergence

    def get_config(self):
        config = {'units': self.units,
                  'dim': self.dim,
                  'gamma': self.gamma,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
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
        return (output_shape, (1,))  # ((n_nodes, output_dim), (1,))


class GaussionConvolution_D(Layer):
    """
        Robust graph convolution layer as in: 
        `Robust Graph Convolutional Networks Against Adversarial Attacks` (https://dl.acm.org/doi/10.1145/3292500.3330851)
        Tensorflow 1.x implementation: https://github.com/thumanlab/nrlweb/blob/master/static/assets/download/RGCN.zip

        `GaussionConvolution_F` implements the GaussionConvolution operation
           where the inputs is node feature distribution (concatenated `mean vector` and `variance vector`) 
           and two adjacency matrices, the output is a projected feature matrix sampled from Gaussian distribution.

        Arguments:
          units: Positive integer, dimensionality of the output space.
          gamma: float scalar, decide the attention weights for mean and variance. 
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
          tuple/list with three 2-D tensor: Tensor `x` and SparseTensor `adj_0, adj_1`: 
            `[(n_nodes, n_features), (n_nodes, n_nodes), (n_nodes, n_nodes)]`.
          The former one is the feature matrix (Tensor) and the others are adjacency matrix (SparseTensor) with different normalize rate (-0.5, -1.0).

        Output shape:
          2-D tensor with shape: `(n_nodes, units)` which is sampled from Gaussian distribution.
    """

    def __init__(self, units,
                 gamma=1.,
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
        self.gamma = gamma
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
        feature_shape = input_shapes[0]

        self.dim = feature_shape[1] // 2

        self.kernel_mean = self.add_weight(shape=(self.dim, self.units),
                                           initializer=self.kernel_initializer,
                                           name='kernel_mean',
                                           regularizer=self.kernel_regularizer,
                                           constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias_mean = self.add_weight(shape=(self.units,),
                                             initializer=self.bias_initializer,
                                             name='bias_mean',
                                             regularizer=self.bias_regularizer,
                                             constraint=self.bias_constraint)
        else:
            self.bias = None

        self.kernel_var = self.add_weight(shape=(self.dim, self.units),
                                          initializer=self.kernel_initializer,
                                          name='kernel_var',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias_var = self.add_weight(shape=(self.units,),
                                            initializer=self.bias_initializer,
                                            name='bias_var',
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)
        else:
            self.bias = None

        super().build(input_shapes)

    def call(self, inputs):
        x, *adj = inputs
#         assert len(adj) == 2

        mean = tf.slice(x, [0, 0], [-1, self.dim])
        var = tf.slice(x, [0, self.dim], [-1, self.dim])

        mean = activations.elu(mean @ self.kernel_mean)
        var = activations.relu(var @ self.kernel_var)

        attention = tf.math.exp(-self.gamma*var)
        mean = tf.sparse.sparse_dense_matmul(adj[0], mean * attention)
        var = tf.sparse.sparse_dense_matmul(adj[1], var * attention * attention)

        if self.use_bias:
            mean += self.bias_mean
            var += self.bias_var

        sample = tf.random.normal(tf.shape(var), 0, 1, dtype=config.floatx())
        output = mean + tf.math.sqrt(var + 1e-8) * sample
        return self.activation(output)

    def get_config(self):
        config = {'units': self.units,
                  'gamma': self.gamma,
                  'dim': self.dim,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
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
