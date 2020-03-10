from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras.layers import Layer, Dropout, LeakyReLU

import tensorflow as tf


class ChebyConvolution(Layer):
    """Basic graph convolution layer as in https://arxiv.org/abs/1609.02907"""
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
            
        self.built = True
        super().build(input_shapes)
        
        
    def call(self, inputs):
        
        features, adjs = inputs
        supports = []
        for adj, kernel, bias in zip(adjs, self.kernel, self.bias):
            support = features @ kernel
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
        return output_shape  # (batch_size, output_dim)
