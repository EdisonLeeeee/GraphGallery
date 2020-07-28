from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras.layers import Layer, Dropout, LeakyReLU

import tensorflow as tf


class GraphAttention(Layer):
    """
        Basic graph attention layer as in: 
        [Graph Attention Networks](https://arxiv.org/abs/1710.10903)
        Tensorflow 1.x implementation: https://github.com/PetarV-/GAT
        Pytorch implementation: https://github.com/Diego999/pyGAT
        Keras implementation: https://github.com/danielegrattarola/keras-gat


        Arguments:
          units: Positive integer, dimensionality of the output space.
          atten_heads: Positive integer, number of attention heads.
          attn_heads_reduction: {'concat', 'average'}, whether to enforce concat or average for the outputs from different heads.
          dropout: Float, internal dropout rate
          activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
          use_bias: Boolean, whether the layer uses a bias vector.
          kernel_initializer: Initializer for the `kernel` weights matrix.
          attn_kernel_initializer: Initializer for the `attn_kernel` weights matrix.
          bias_initializer: Initializer for the bias vector.
          kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
          attn_kernel_regularizer: Regularizer function applied to
            the `attn_kernel` weights matrix.
          bias_regularizer: Regularizer function applied to the bias vector.
          activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
          kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
          attn_kernel_constraint: Constraint function applied to
            the `attn_kernel` weights matrix.
          bias_constraint: Constraint function applied to the bias vector.

        Input shape:
          tuple/list with two 2-D tensor: Tensor `x` and SparseTensor `adj`: `[(n_nodes, n_features), (n_nodes, n_nodes)]`.
          The former one is the feature matrix (Tensor) and the last is adjacency matrix (SparseTensor).

        Output shape:
          2-D tensor with shape: `(n_nodes, units)` (use average) or `(n_nodes, attn_heads * units)` (use concat).       
    """

    def __init__(self,
                 units,
                 attn_heads=1,
                 attn_heads_reduction='concat',
                 dropout=0.5,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 **kwargs):

        super().__init__(**kwargs)

        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        self.units = units  # Number of output features (F' in the paper)
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # Eq. 5 and 6 in the paper
        self.dropout = dropout  # Internal dropout rate
        self.activation = activations.get(activation)  # Eq. 4 in the paper
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)

        # Populated by build()
        self.kernels = []       # Layer kernels for attention heads
        self.biases = []        # Layer biases for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads

        if attn_heads_reduction == 'concat':
            # Output will have shape (..., K * F')
            self.output_dim = self.units * self.attn_heads
        else:
            # Output will have shape (..., F')
            self.output_dim = self.units

    def build(self, input_shape):
        input_dim = input_shape[0][-1]

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     name=f'kernel_{head}')
            self.kernels.append(kernel)

            # Layer bias
            if self.use_bias:
                bias = self.add_weight(shape=(self.units, ),
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint,
                                       name=f'bias_{head}')
                self.biases.append(bias)

            # Attention kernels
            attn_kernel_self = self.add_weight(shape=(self.units, 1),
                                               initializer=self.attn_kernel_initializer,
                                               regularizer=self.attn_kernel_regularizer,
                                               constraint=self.attn_kernel_constraint,
                                               name=f'attn_kernel_self_{head}')
            attn_kernel_neighs = self.add_weight(shape=(self.units, 1),
                                                 initializer=self.attn_kernel_initializer,
                                                 regularizer=self.attn_kernel_regularizer,
                                                 constraint=self.attn_kernel_constraint,
                                                 name=f'attn_kernel_neigh_{head}')
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])

        super().build(input_shape)

    def call(self, inputs):
        '''
        inputs: (x, adj), x is node feature matrix with shape [N, F], 
        adj is adjacency matrix with shape [N, N].

        Note:
        N: number of nodes
        F: input dim
        F': output dim
        '''
        x, adj = inputs

        outputs = []
        for head in range(self.attn_heads):
            kernel = self.kernels[head]  # W in the paper
            attn_kernel_self, attn_kernel_neighs = self.attn_kernels[head]  # Attention kernel a in the paper (2F' x 1)

            # Compute inputs to attention network
            h = x @ kernel  # [N, F']

            # Compute attentions for self and neighbors
            attn_for_self = h @ attn_kernel_self  # [N, 1]
            attn_for_neighs = h @ attn_kernel_neighs  # [N, 1]

            # combine the attention with adjacency matrix via broadcast
            attn_for_self = adj * attn_for_self
            attn_for_neighs = adj * tf.transpose(attn_for_neighs)

            # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
            attentions = tf.sparse.add(attn_for_self, attn_for_neighs)

            # Add nonlinearty by LeakyReLU
            attentions = tf.sparse.SparseTensor(indices=attentions.indices,
                                                values=LeakyReLU(alpha=0.2)(attentions.values),
                                                dense_shape=attentions.dense_shape
                                                )
            # Apply softmax to get attention coefficients
            attentions = tf.sparse.softmax(attentions)  # (N x N)

            # Apply dropout to features and attention coefficients
            if self.dropout:
                attentions = tf.sparse.SparseTensor(indices=attentions.indices,
                                                    values=Dropout(rate=self.dropout)(attentions.values),
                                                    dense_shape=attentions.dense_shape
                                                    )  # (N x N)
                h = Dropout(self.dropout)(h)  # (N x F')

            # Linear combination with neighbors' features
            h = tf.sparse.sparse_dense_matmul(attentions, h)  # (N x F')

            if self.use_bias:
                h += self.biases[head]

            # Add output of attention head to final output
            outputs.append(h)

        # Aggregate the heads' output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output = tf.concat(outputs, axis=1)  # (N x KF')
        else:
            output = tf.reduce_mean(tf.stack(outputs), axis=0)  # (N x F')

        return self.activation(output)

    def get_config(self):

        config = {'units': self.units,
                  'attn_heads': self.attn_heads,
                  'attn_heads_reduction': self.attn_heads_reduction,
                  'use_bias': self.use_bias,
                  'activation': keras.activations.serialize(self.activation),
                  'kernel_initializer': keras.initializers.serialize(
                      self.kernel_initializer),
                  'attn_kernel_initializer': keras.initializers.serialize(
                      self.attn_kernel_initializer),
                  'bias_initializer': keras.initializers.serialize(
                      self.bias_initializer),
                  'kernel_regularizer': keras.regularizers.serialize(
                      self.kernel_regularizer),
                  'attn_kernel_constraint': keras.regularizers.serialize(
                      self.attn_kernel_constraint),
                  'bias_regularizer': keras.regularizers.serialize(
                      self.bias_regularizer),
                  'activity_regularizer': keras.regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': keras.constraints.serialize(
                      self.kernel_constraint),
                  'attn_kernel_constraint': keras.constraints.serialize(
                      self.attn_kernel_constraint),
                  'bias_constraint': keras.constraints.serialize(self.bias_constraint)
                  }

        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][0], self.output_dim
        return tf.TensorShape(output_shape)
