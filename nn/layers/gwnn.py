from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras.layers import Layer, Dropout, LeakyReLU

import tensorflow as tf
import numpy as np

class WaveletConvolution(Layer):
    """Basic graph convolution layer as in https://arxiv.org/abs/1609.02907"""
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
            
        self.built = True
        super().build(input_shapes)
        
        
    def call(self, inputs):
        
        features, wavelet, inverse_wavelet = inputs
        transformed_features = features @ self.kernel
        
        output = tf.sparse.sparse_dense_matmul(inverse_wavelet, transformed_features)
        filter_ = tf.sparse.SparseTensor(indices=self.indices, 
                                         values=self.filter, 
                                         dense_shape=self.indices[-1]+1)

        output =  tf.sparse.sparse_dense_matmul(filter_, output)
        output = tf.sparse.sparse_dense_matmul(wavelet, output)

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
        return output_shape  # (batch_size, output_dim)
