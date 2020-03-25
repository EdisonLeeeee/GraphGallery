from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras.layers import Layer, Dropout, LeakyReLU

import tensorflow as tf

class GraphAttention(Layer):

    def __init__(self,
                 F_,
                 attn_heads=1,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
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

        self.F_ = F_  # Number of output features (F' in the paper)
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
            self.output_dim = self.F_ * self.attn_heads
        else:
            # Output will have shape (..., F')
            self.output_dim = self.F_


    def build(self, input_shape):
        F = input_shape[0][-1]

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(shape=(F, self.F_),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     name=f'kernel_{head}')
            self.kernels.append(kernel)

            # # Layer bias
            if self.use_bias:
                bias = self.add_weight(shape=(self.F_, ),
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint,
                                       name=f'bias_{head}')
                self.biases.append(bias)

            # Attention kernels
            attn_kernel_self = self.add_weight(shape=(self.F_, 1),
                                               initializer=self.attn_kernel_initializer,
                                               regularizer=self.attn_kernel_regularizer,
                                               constraint=self.attn_kernel_constraint,
                                               name=f'attn_kernel_self_{head}')
            attn_kernel_neighs = self.add_weight(shape=(self.F_, 1),
                                                 initializer=self.attn_kernel_initializer,
                                                 regularizer=self.attn_kernel_regularizer,
                                                 constraint=self.attn_kernel_constraint,
                                                 name=f'attn_kernel_neigh_{head}')
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])
            
        self.built = True
        super().build(input_shape)

    def call(self, inputs):
        X, adj = inputs  # Node features (N x F), Adjacency matrix (N x N)

        outputs = []
        for head in range(self.attn_heads):
            kernel = self.kernels[head]  # W in the paper (F x F')
            attn_kernel_self, attn_kernel_neighs = self.attn_kernels[head]  # Attention kernel a in the paper (2F' x 1)

            # Compute inputs to attention network
            features = X @ kernel  # (N x F')

            # Compute feature combinations
            # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            attn_for_self = features @ attn_kernel_self    # (N x 1), [a_1]^T [Wh_i]
            attn_for_neighs = features @ attn_kernel_neighs  # (N x 1), [a_2]^T [Wh_j]
            
            
            attn_for_self = adj * attn_for_self
            attn_for_neighs = adj * tf.transpose(attn_for_neighs)
            
            
            # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
            attentions = tf.sparse.add(attn_for_self, attn_for_neighs) 

            # Add nonlinearty
            attentions = tf.sparse.SparseTensor(indices=attentions.indices,
                                                values=LeakyReLU(alpha=0.2)(attentions.values),
                                                dense_shape=attentions.dense_shape
                                               )
            # Apply softmax to get attention coefficients
            attentions = tf.sparse.softmax(attentions) # (N x N)

            # Apply dropout to features and attention coefficients
            if self.dropout:
                attentions = tf.sparse.SparseTensor(indices=attentions.indices,
                                                values=Dropout(rate=self.dropout)(attentions.values),
                                                dense_shape=attentions.dense_shape
                                               )  # (N x N)
                features = Dropout(self.dropout)(features)  # (N x F')

            # Linear combination with neighbors' features
            node_features = tf.sparse.sparse_dense_matmul(attentions, features)  # (N x F')

            if self.use_bias:
                node_features += self.biases[head]

            # Add output of attention head to final output
            outputs.append(node_features)

        # Aggregate the heads' output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output = tf.concat(outputs, axis=1)  # (N x KF')
        else:
            output = tf.reduce_mean(tf.stack(outputs), axis=0)  # (N x F')

        return self.activation(output)

    def get_config(self):

        config = {'F_': self.F_,
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
                  'attn_kernel_constraint':keras.regularizers.serialize(
                      self.attn_kernel_constraint),
                  'bias_regularizer': keras.regularizers.serialize(
                      self.bias_regularizer),
                  'activity_regularizer': keras.regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': keras.constraints.serialize(
                      self.kernel_constraint),
                  'attn_kernel_constraint':keras.constraints.serialize(
                      self.attn_kernel_constraint),
                  'bias_constraint': keras.constraints.serialize(self.bias_constraint)
                 }

        base_config = super().get_config()
        return {**base_config, **config}
    
    
    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][0], self.output_dim
        return output_shape
