import tensorflow as tf
from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras.layers import Layer, Dropout, Conv1D


class LGConvolution(Layer):
    def __init__(self, filters, kernel_size,
                 use_bias=False,
                 dropout=0.5,
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
        self.filters = filters
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.dropout_rate = dropout
        
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)


    def build(self, input_shapes):
        
        kernel_size = self.kernel_size
        filters = self.filters
        F = input_shapes[-1]
        
        self.dropout = Dropout(rate=self.dropout_rate)
        self.conv1 = Conv1D((F+filters)//2, (kernel_size+1)//2+1, 
                            use_bias=True,
                            activation=self.activation,
                            kernel_initializer=self.kernel_initializer,
                            bias_initializer=self.bias_initializer,
                            kernel_regularizer=self.kernel_regularizer,
                            bias_regularizer=self.bias_regularizer,
                            kernel_constraint=self.kernel_constraint,
                            bias_constraint=self.bias_constraint,                               
                            name='conv1')
        self.conv2 = Conv1D(filters, kernel_size//2+1, 
                            use_bias=self.use_bias,
                            activation=self.activation,
                            kernel_initializer=self.kernel_initializer,
                            bias_initializer=self.bias_initializer,
                            kernel_regularizer=self.kernel_regularizer,
                            bias_regularizer=self.bias_regularizer,
                            kernel_constraint=self.kernel_constraint,
                            bias_constraint=self.bias_constraint,                                
                            name='conv2')        
                                                             
        
        self.built = True
        super().build(input_shapes)
        
        
    def call(self, inputs):
        
        h = inputs
        h = self.dropout(h)
        h = self.conv1(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = tf.squeeze(h, axis=1)
        return h
            

    def get_config(self):
        config = {'filters': self.filters,
                  'kernel_size': self.kernel_size,
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
        output_shape = (input_shapes[0], self.filters)
        return output_shape  # (batch_size, output_dim)