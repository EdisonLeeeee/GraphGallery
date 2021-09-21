from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras.layers import Layer

import tensorflow as tf


class MedianAggregator(Layer):
    """
        Basic graphSAGE convolution layer as in: 
        [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)
        Tensorflow 1.x implementation: https://github.com/williamleif/GraphSAGE
        Pytorch implementation: https://github.com/williamleif/graphsage-simple/

        Aggregates via mean followed by matmul and non-linearity.

        `MedianAggregator` implements the operation:
        `output = activation(Concat(x @ kernel_0, Median(neigh_x) @ kernel_1) + bias)`
        where `x` is the node attribute matrix, `neigh_x` is the node attribute matrix of neighbors,
        `Agg` is the operation of aggregation (`mean`, `sum`, `max`, `min`) along the last dimension,
        `Concat` is the operation of concatenation between transformed node attributes and neighbor attributes,
        and it could be replaced with `Add` operation.
        `activation` is the element-wise activation function
        passed as the `activation` argument, `kernel` is a weights matrix
        created by the layer, and `bias` is a bias vector created by the layer
        (only applicable if `use_bias` is `True`).


        Parameters:
          units: Positive integer, dimensionality of the output space.
          concat: bool, whether the layer uses concatenation 
            between transformed node attributes and neighbor attributes, 
            if `False`, the `Add` operation will be used.
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

        Input shape:
          tuple/list with two tensor: 2-D Tensor `x` and 3-D Tensor `neigh_x`: 
          `[(batch_num_nodes, num_node_attrs), (batch_num_nodes, sizes, num_node_attrs)]`.
          The former one is the node attribute matrix (Tensor) and the last is the neighbor node attribute matrix (Tensor).

        Output shape:
          2-D tensor with shape: `(batch_num_nodes, units)` or `(batch_num_nodes, units * 2)`,
          depend on using `Add` or `Concat` for the node and neighbor attributes.       
    """

    def __init__(self,
                 units,
                 concat=False,
                 use_bias=True,
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
        self.concat = concat
        self.use_bias = use_bias
        self.activation = activations.get(activation)

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        if concat:
            self.output_dim = units * 2
        else:
            self.output_dim = units

    def build(self, input_shape):
        input_dim = input_shape[0][-1]

        self.kernel_self = self.add_weight(shape=(input_dim, self.units),
                                           initializer=self.kernel_initializer,
                                           regularizer=self.kernel_regularizer,
                                           constraint=self.kernel_constraint,
                                           name='kernel_self')
        self.kernel_neigh = self.add_weight(shape=(input_dim, self.units),
                                            initializer=self.kernel_initializer,
                                            regularizer=self.kernel_regularizer,
                                            constraint=self.kernel_constraint,
                                            name='kernel_neigh')

        # Layer bias
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim, ),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        name='bias')

        super().build(input_shape)

    def call(self, inputs):
        x, neigh_x = inputs
        neigh_x = tf.transpose(neigh_x, perm=[0, 2, 1])
        n = neigh_x.shape[-1]
        n = n // 2 + tf.math.mod(n, 2)
        neigh_x = tf.raw_ops.NthElement(input=neigh_x, n=n)

        x = x @ self.kernel_self
        neigh_x = neigh_x @ self.kernel_neigh

        if self.concat:
            output = tf.concat([x, neigh_x], axis=1)
        else:
            output = x + neigh_x

        if self.use_bias:
            output += self.bias

        return self.activation(output)

    def get_config(self):

        config = {'units': self.units,
                  'concat': self.concat,
                  'use_bias': self.use_bias,
                  'activation': keras.activations.serialize(self.activation),
                  'kernel_initializer': keras.initializers.serialize(
                      self.kernel_initializer),
                  'bias_initializer': keras.initializers.serialize(
                      self.bias_initializer),
                  'kernel_regularizer': keras.regularizers.serialize(
                      self.kernel_regularizer),
                  'bias_regularizer': keras.regularizers.serialize(
                      self.bias_regularizer),
                  'activity_regularizer': keras.regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': keras.constraints.serialize(
                      self.kernel_constraint),
                  'bias_constraint': keras.constraints.serialize(self.bias_constraint)
                  }

        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][0], self.output_dim
        return output_shape  # (batch_num_nodes, units) or (batch_num_nodes, units * 2)


class MedianGCNAggregator(Layer):
    """
        Basic graphSAGE convolution layer as in: 
        [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)
        Tensorflow 1.x implementation: https://github.com/williamleif/GraphSAGE
        Pytorch implementation: https://github.com/williamleif/graphsage-simple/

        Aggregates via mean followed by matmul and non-linearity.
        Same matmul parameters are used self vector and neighbor vectors.    

        `GCNAggregator` implements the operation:
        `output = activation(Agg(Concat(neigh_x, x)) @ kernel) + bias)`
        where `x` is the node attribute matrix, `neigh_x` is the node attribute matrix of neighbors,
        `Agg` is the operation of aggregation (`mean`, `sum`, `max`, `min`) along the last dimension,
        `activation` is the element-wise activation function
        passed as the `activation` argument, `kernel` is a weights matrix
        created by the layer, and `bias` is a bias vector created by the layer
        (only applicable if `use_bias` is `True`).


        Parameters:
          units: Positive integer, dimensionality of the output space.
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

        Input shape:
          tuple/list with two tensor: 2-D Tensor `x` and 3-D Tensor `neigh_x`: 
          `[(batch_num_nodes, num_node_attrs), (batch_num_nodes, sizes, num_node_attrs)]`.
          The former one is the node attribute matrix (Tensor) and the last is the neighbor node attribute matrix (Tensor).

        Output shape:
          2-D tensor with shape: `(batch_num_nodes, units)` or `(batch_num_nodes, units * 2)`,
          depend on using `Add` or `Concat` for the node and neighbor attributes.       
    """

    def __init__(self,
                 units,
                 use_bias=True,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        kwargs.pop('concat', None)  # in order to be compatible with `SAGEAggregator`
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

    def build(self, input_shape):
        input_dim = input_shape[0][-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      name='kernel')

        # Layer bias
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units, ),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        name='bias')

        super().build(input_shape)

    def call(self, inputs):

        x, neigh_x = inputs
        x = tf.expand_dims(x, axis=1)
        agg = tf.concat([x, neigh_x], axis=1)

        h = tf.transpose(agg, perm=[0, 2, 1])
        n = h.shape[-1]
        n = n // 2
        h = tf.raw_ops.NthElement(input=h, n=n)

        output = h @ self.kernel

        if self.use_bias:
            output += self.bias

        return self.activation(output)

    def get_config(self):

        config = {'units': self.units,
                  'use_bias': self.use_bias,
                  'activation': keras.activations.serialize(self.activation),
                  'kernel_initializer': keras.initializers.serialize(
                      self.kernel_initializer),
                  'bias_initializer': keras.initializers.serialize(
                      self.bias_initializer),
                  'kernel_regularizer': keras.regularizers.serialize(
                      self.kernel_regularizer),
                  'bias_regularizer': keras.regularizers.serialize(
                      self.bias_regularizer),
                  'activity_regularizer': keras.regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': keras.constraints.serialize(
                      self.kernel_constraint),
                  'bias_constraint': keras.constraints.serialize(self.bias_constraint)
                  }

        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][0], self.units
        return tf.TensorShape(output_shape)  # (batch_num_nodes, units) or (batch_num_nodes, units * 2)
