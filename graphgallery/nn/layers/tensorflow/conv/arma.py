from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Layer


class ARMAConv(Layer):
    r"""
    An Auto-Regressive Moving Average convolutional layer (ARMA) from the paper

    [Graph Neural Networks with convolutional ARMA filters](https://arxiv.org/abs/1901.01343)

    **Mode**: single, disjoint, mixed, batch.

    This layer computes:
    $$
        \X' = \frac{1}{K} \sum\limits_{k=1}^K \bar\X_k^{(T)},
    $$
    where \(K\) is the order of the ARMA\(_K\) filter, and where:
    $$
        \bar \X_k^{(t + 1)} =
        \sigma \left(\tilde \A \bar \X^{(t)} \W^{(t)} + \X \V^{(t)} \right)
    $$
    is a recursive approximation of an ARMA\(_1\) filter, where
    \( \bar \X^{(0)} = \X \)
    and
    $$
        \tilde \A =  \D^{-1/2} \A \D^{-1/2}.
    $$

    **Input**

    - Node features of shape `([batch], n_nodes, n_node_features)`;
    - Normalized and rescaled Laplacian of shape `([batch], n_nodes, n_nodes)`; can be
    computed with `spektral.utils.convolution.normalized_laplacian` and
    `spektral.utils.convolution.rescale_laplacian`.

    **Output**

    - Node features with the same shape as the input, but with the last
    dimension changed to `units`.

    Parameters:
    - `units`: number of output units;
    - `order`: order of the full ARMA\(_K\) filter, i.e., the number of parallel
    stacks in the layer;
    - `iterations`: number of iterations to compute each ARMA\(_1\) approximation;
    - `share_weights`: share the weights in each ARMA\(_1\) stack.
    - `gcn_activation`: gcn_activation function to compute each ARMA\(_1\)
    stack;
    - `dropout_rate`: dropout rate for skip connection;
    - `activation`: activation function;
    - `use_bias`: bool, add a bias vector to the output;
    - `kernel_initializer`: initializer for the weights;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the weights;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the weights;
    - `bias_constraint`: constraint applied to the bias vector.
    """

    def __init__(
        self,
        units,
        order=1,
        iterations=1,
        share_weights=False,
        gcn_activation="elu",
        activation=None,
        dropout=0.75,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.iterations = iterations
        self.order = order
        self.share_weights = share_weights
        self.use_bias = use_bias

        self.activation = activations.get(activation)
        self.gcn_activation = activations.get(gcn_activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.dropout = Dropout(dropout)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        F = input_shape[0][-1]

        # Create weights for parallel stacks
        # self.kernels[k][i] refers to the k-th stack, i-th iteration
        self.kernels = []
        for k in range(self.order):
            kernel_stack = []
            current_shape = F
            for i in range(self.iterations):
                kernel_stack.append(
                    self.create_weights(
                        current_shape, F, self.units, "ARMA_GCS_{}{}".format(k, i)
                    )
                )
                current_shape = self.units
                if self.share_weights and i == 1:
                    # No need to continue because all weights will be shared
                    break
            self.kernels.append(kernel_stack)

        self.built = True

    def call(self, inputs):
        x, a = inputs

        output = []
        for k in range(self.order):
            output_k = x
            for i in range(self.iterations):
                output_k = self.gcs([output_k, x, a], k, i)
            output.append(output_k)
        output = K.stack(output, axis=-1)
        output = K.mean(output, axis=-1)
        output = self.activation(output)

        return output

    def create_weights(self, input_dim, input_dim_skip, units, name):
        """
        Creates a set of weights for a GCN with skip connections.
        :param input_dim: dimension of the input space
        :param input_dim_skip: dimension of the input space for the skip connection
        :param units: dimension of the output space
        :param name: name of the layer
        :return:
            - kernel_1, from input space of the layer to output space
            - kernel_2, from input space of the skip connection to output space
            - bias, bias vector on the output space if use_bias=True, None otherwise.
        """
        kernel_1 = self.add_weight(
            shape=(input_dim, units),
            name=name + "_kernel_1",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        kernel_2 = self.add_weight(
            shape=(input_dim_skip, units),
            name=name + "_kernel_2",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        bias = None
        if self.use_bias:
            bias = self.add_weight(
                shape=(units,),
                name=name + "_bias",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        return kernel_1, kernel_2, bias

    def gcs(self, inputs, stack, iteration):
        """
        Creates a graph convolutional layer with a skip connection.
        :param inputs: list of input Tensors, namely
            - input node features
            - input node features for the skip connection
            - normalized adjacency matrix;
        :param stack: int, current stack (used to retrieve kernels);
        :param iteration: int, current iteration (used to retrieve kernels);
        :return: output node features.
        """
        x, x_skip, a = inputs

        itr = 1 if self.share_weights and iteration >= 1 else iteration
        kernel_1, kernel_2, bias = self.kernels[stack][itr]

        output = K.dot(x, kernel_1)
        output = K.dot(a, output)

        skip = K.dot(x_skip, kernel_2)
        skip = self.dropout(skip)
        output += skip

        if self.use_bias:
            output = K.bias_add(output, bias)
        output = self.gcn_activation(output)

        return output

    def get_config(self):
        config = {
            "units": self.units,
            "iterations": self.iterations,
            "order": self.order,
            "share_weights": self.share_weights,
            "activation": activations.serialize(self.activation),
            "gcn_activation": activations.serialize(self.gcn_activation),
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
            'bias_constraint': constraints.serialize(self.bias_constraint),
        }
        base_config = super().get_config()
        return {**base_config, **config}
