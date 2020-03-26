
from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras.layers import Layer, Dropout, LeakyReLU

import tensorflow as tf

class GaussionConvolution_F(Layer):
    '''The input is features'''    
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
        self.built = True
        super().build(input_shapes)

    def call(self, inputs):
        features, *adj = inputs
        assert len(adj) == 2
        
        features = features @ self.kernel

        if self.use_bias:
            features += self.bias
            
        mean = activations.elu(tf.slice(features, [0, 0], [-1, self.dim]))
        var = activations.relu(tf.slice(features, [0, self.dim], [-1, self.dim]))
        
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
        return  (output_shape, (1,))  # (batch_size, output_dim)
    
class GaussionConvolution_D(Layer):
    '''The input is distribution'''    
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
        self.built = True
        super().build(input_shapes)
    
    def call(self, inputs):
        features, *adj = inputs
        assert len(adj) == 2

        mean = tf.slice(features, [0, 0], [-1, self.dim])
        var = tf.slice(features, [0, self.dim], [-1, self.dim])
        
        mean = activations.elu(mean @ self.kernel_mean)
        var = activations.relu(var @ self.kernel_var)
        
        attention = tf.math.exp(-self.gamma*var)
        mean = tf.sparse.sparse_dense_matmul(adj[0], mean * attention)
        var = tf.sparse.sparse_dense_matmul(adj[1], var * attention * attention)
        
        if self.use_bias:
            mean += self.bias_mean
            var += self.bias_var
            
        sample = tf.random.normal(tf.shape(var), 0, 1, dtype=tf.float32)
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
        return output_shape  # (batch_size, output_dim)
