import tensorflow as tf
from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras.layers import Layer, Dropout, Conv1D


class LGConvolution(Layer):
    """
        Large-scale graph convolution layer as in:
        [Large-Scale Learnable Graph Convolutional Networks](https://arxiv.org/abs/1808.03965)
        Tensorflow 1.x implementation: https://github.com/divelab/lgcn

        `LGConvolution` implements the operation:
        `output = Conv1d(Conv1d(x))`, where `x` is the input feature matrix, 
        and the dropout will be used in `x` and hidden outpus.


        Arguments:
            filters: Integer, related to the dimensionality of the output space
              (i.e. the number of output filters in the convolution).
            kernel_size: An integer, related to the
              height and width of the 2D convolution window.
              Can be a single integer to specify the same value for
              all spatial dimensions.              
            dropout: Float, the dropout rate of inputs and hidden outputs.
            use_bias: Boolean, whether the layer uses a bias vector.
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
            2-D tensor: Tensor `x`: `(n_nodes, n_features)`. where `x` is the node feature matrix (Tensor).

        Output shape:
            2-D tensor with shape: `(n_nodes, filters)`.
    """

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
        input_dim = input_shapes[-1]

        self.dropout = Dropout(rate=self.dropout_rate)
        self.conv1 = Conv1D((input_dim+filters)//2, (kernel_size+1)//2+1,
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
        return tf.TensorShape(output_shape)  # (n_nodes, output_dim)
