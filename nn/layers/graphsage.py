from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras.layers import Layer

import tensorflow as tf

class SAGEConvolution(Layer):

    def __init__(self,
                 units,
                 concat=False,
                 use_bias=True,
                 agg_method='mean',
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
        self.agg_method = agg_method
        self.aggregator = {'mean':tf.reduce_mean, 'sum':tf.reduce_sum,
                    'max':tf.reduce_max, 'min':tf.reduce_min}[agg_method]
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

        # # Layer bias
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim, ),
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint,
                                   name='bias')

        self.built = True
        super().build(input_shape)

    def call(self, inputs):
        node_feat, neigh_feat = inputs
        neigh_feat = self.aggregator(neigh_feat, axis=1)
        
        node_feat = node_feat @ self.kernel_self
        neigh_feat = neigh_feat @ self.kernel_neigh
        
        if self.concat:
            output = tf.concat([node_feat, neigh_feat], axis=1)
        else:
            output = node_feat + neigh_feat

        if self.use_bias:
            output += self.bias

        return self.activation(output)

    def get_config(self):

        config = {'units': self.units,
                  'concat': self.concat,
                  'use_bias': self.use_bias,
                  'agg_method': self.agg_method,
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
        return output_shape
