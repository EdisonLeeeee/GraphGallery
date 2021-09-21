
from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras.layers import Layer, Dropout, LeakyReLU

import tensorflow as tf


class GaussionConvF(Layer):
    """
        Robust graph convolution layer as in: 
        `Robust Graph Convolutional Networks Against Adversarial Attacks` <https://dl.acm.org/doi/10.1145/3292500.3330851>
        Tensorflow 1.x implementation: <https://github.com/thumanlab/nrlweb/blob/master/static/assets/download/RGCN.zip>

        `GaussionConvF` implements the GaussionConvolution operation
           where the inputs is node attribute matrix and two adjacency matrices,
           the output is another distribution (represented with `mean vector` 
           and `variance vector`) 

        Parameters:
          units: Positive integer, dimensionality of the output space.
          gamma: Float scalar, determining the attention weights for mean and variance. 
          use_bias: bool, whether the layer uses a bias vector.
          activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
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

        Inputs:
          tuple/list with three 2-D tensor: Tensor `x` and SparseTensor `adj_0, adj_1`: 
            `[(num_nodes, num_node_attrs), (num_nodes, num_nodes), (num_nodes, num_nodes)]`.
          The former one is the node attribute matrix (Tensor) and the others are adjacency matrix (SparseTensor) with different normalize rate (-0.5, -1.0).

        Outputs:
          shape ((num_nodes, units), (num_nodes, units)), two 2-D tensor representing the
          `mean` and `variance` of outputs.
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
        self.kernel = self.add_weight(shape=(input_shapes[0][1], self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias_mean = self.add_weight(shape=(self.units,),
                                             initializer=self.bias_initializer,
                                             name='bias_mean',
                                             regularizer=self.bias_regularizer,
                                             constraint=self.bias_constraint)
            self.bias_var = self.add_weight(shape=(self.units,),
                                            initializer=self.bias_initializer,
                                            name='bias_var',
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)
        else:
            self.bias_mean = None
            self.bias_val = None

        super().build(input_shapes)

    def call(self, inputs):
        x, *adj = inputs
#         assert len(adj) == 2

        h = x @ self.kernel

        mean = activations.elu(h)
        var = activations.relu(h)

        attention = tf.exp(-self.gamma * var)
        mean = tf.sparse.sparse_dense_matmul(adj[0], mean * attention)
        var = tf.sparse.sparse_dense_matmul(adj[1], var * attention * attention)

        if self.use_bias:
            mean += self.bias_mean
            var += self.bias_var

        return self.activation(mean), self.activation(var)

    def get_config(self):
        config = {'units': self.units,
                  'gamma': self.gamma,
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
        return tf.TensorShape(output_shape), tf.TensorShape(output_shape)


class GaussionConvD(Layer):
    """
        Robust graph convolution layer as in: 
        `Robust Graph Convolutional Networks Against Adversarial Attacks` <https://dl.acm.org/doi/10.1145/3292500.3330851>
        Tensorflow 1.x implementation: <https://github.com/thumanlab/nrlweb/blob/master/static/assets/download/RGCN.zip>

        `GaussionConvF` implements the GaussionConvolution operation
           where the inputs are the distribution (represented with `mean vector` and `variance vector`) 
           and two adjacency matrices, the output is another distribution (represented with `mean vector` 
           and `variance vector`) 

        Parameters:
          units: Positive integer, dimensionality of the output space.
          gamma: Float scalar, decide the attention weights for mean and variance. 
          activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
          use_bias: bool, whether the layer uses a bias vector.
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

        Inputs:
          tuple/list with four 2-D tensor: Tensor `mean`, `var` and SparseTensor `adj_0`, `adj_1`: 
              `[(num_nodes, num_node_attrs), (num_nodes, num_node_attrs), (num_nodes, num_nodes), (num_nodes, num_nodes)]`.
          The former two is the mean and variance vector (Tensor) and the last are adjacency matrices (SparseTensor) with 
              different normalize rates (-0.5, -1.0).

        Outputs:
          Shape ((num_nodes, units), (num_nodes, units)), two 2-D tensors representing the
              `mean` and `var` of outputs.
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
        attribute_shape_mean, attribute_shape_var = input_shapes[:2]

        self.kernel_mean = self.add_weight(shape=(attribute_shape_mean[1], self.units),
                                           initializer=self.kernel_initializer,
                                           name='kernel_mean',
                                           regularizer=self.kernel_regularizer,
                                           constraint=self.kernel_constraint)
        self.kernel_var = self.add_weight(shape=(attribute_shape_var[1], self.units),
                                          initializer=self.kernel_initializer,
                                          name='kernel_var',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias_mean = self.add_weight(shape=(self.units,),
                                             initializer=self.bias_initializer,
                                             name='bias_mean',
                                             regularizer=self.bias_regularizer,
                                             constraint=self.bias_constraint)
            self.bias_var = self.add_weight(shape=(self.units,),
                                            initializer=self.bias_initializer,
                                            name='bias_var',
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)
        else:
            self.bias_mean = None
            self.bias_val = None

        super().build(input_shapes)

    def call(self, inputs):
        mean, var, *adj = inputs
#         assert len(adj) == 2

        mean = activations.elu(mean @ self.kernel_mean)
        var = activations.relu(var @ self.kernel_var)

        attention = tf.math.exp(-self.gamma * var)
        mean = tf.sparse.sparse_dense_matmul(adj[0], mean * attention)
        var = tf.sparse.sparse_dense_matmul(adj[1], var * attention * attention)

        if self.use_bias:
            mean += self.bias_mean
            var += self.bias_var

        return self.activation(mean), self.activation(var)

    def get_config(self):
        config = {'units': self.units,
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
        attribute_shape_mean, attribute_shape_var = input_shapes[:2]

        output_shape_mean = (attribute_shape_mean[0], self.units)
        output_shape_var = (attribute_shape_var[1], self.units)

        return tf.TensorShape(output_shape_mean), tf.TensorShape(output_shape_var)
