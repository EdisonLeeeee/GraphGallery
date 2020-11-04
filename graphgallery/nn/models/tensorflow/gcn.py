from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery.nn.layers.tensorflow import GraphConvolution, Gather
from graphgallery import floatx, intx


class GCN(Model):

    def __init__(self, in_channels, out_channels,
                 hiddens=[16],
                 activations=['relu'],
                 dropout=0.5,
                 l2_norm=5e-4,
                 lr=0.01, use_bias=False,
                 experimental_run_tf_function=True):

        x = Input(batch_shape=[None, in_channels],
                  dtype=floatx(), name='attr_matrix')
        adj = Input(batch_shape=[None, None], dtype=floatx(),
                    sparse=True, name='adj_matrix')
        index = Input(batch_shape=[None], dtype=intx(), name='node_index')

        h = x
        for hidden, activation in zip(hiddens, activations):
            h = GraphConvolution(hidden, use_bias=use_bias,
                                 activation=activation,
                                 kernel_regularizer=regularizers.l2(l2_norm))([h, adj])

            h = Dropout(rate=dropout)(h)

        h = GraphConvolution(out_channels, use_bias=use_bias)([h, adj])
        h = Gather()([h, index])

        super().__init__(inputs=[x, adj, index], outputs=h)
        self.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                     optimizer=Adam(lr=lr), metrics=['accuracy'],
                     experimental_run_tf_function=experimental_run_tf_function)

# class GCN(Model):

#     def __init__(self, hiddens,
#                  out_channels, activations=['relu'],
#                  l2_norm=5e-4, dropout=0.5,
#                  lr=0.01, use_bias=False):

#         super().__init__()

#         self.GNN_layers = []
#         for hidden, activation, l2_norm in zip(hiddens, activations, l2_norms):
#             layer = GraphConvolution(hidden, use_bias=use_bias,
#                                          activation=activation,
#                                          kernel_regularizer=regularizers.l2(l2_norm))

#             self.GNN_layers.append(layer)

#         layer = GraphConvolution(out_channels, use_bias=use_bias)
#         self.GNN_layers.append(layer)

#         self.dropout = Dropout(dropout)
#         self.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
#                       optimizer=Adam(lr=lr), metrics=['accuracy'])

#         self.metrics_fn = SparseCategoricalAccuracy()

#     def call(self, inputs, training=False):
#         x, adj, idx = inputs

#         for layer in self.GNN_layers:
#             x = self.dropout(x, training=training)
#             x = layer([x, adj])

#         return tf.gather(x, idx)
