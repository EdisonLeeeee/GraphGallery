from tensorflow.keras import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy


from graphgallery.nn.layers.tensorflow import GCNConv
from graphgallery import floatx
from graphgallery.nn.models.tf_keras import TFKeras


class GCN(TFKeras):

    def __init__(self, in_features, out_features,
                 hids=[16],
                 acts=['relu'],
                 dropout=0.5,
                 weight_decay=5e-4,
                 lr=0.01, bias=False,
                 experimental_run_tf_function=True):

        x = Input(batch_shape=[None, in_features],
                  dtype=floatx(), name='node_attr')
        adj = Input(batch_shape=[None, None], dtype=floatx(),
                    sparse=True, name='adj_matrix')

        h = x
        for hid, act in zip(hids, acts):
            h = GCNConv(hid, use_bias=bias,
                        activation=act,
                        kernel_regularizer=regularizers.l2(weight_decay))([h, adj])

            h = Dropout(rate=dropout)(h)

        h = GCNConv(out_features, use_bias=bias)([h, adj])

        super().__init__(inputs=[x, adj], outputs=h)
        self.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                     optimizer=Adam(lr), metrics=['accuracy'],
                     experimental_run_tf_function=experimental_run_tf_function)

# class GCN(Model):

#     def __init__(self, hids,
#                  out_features, acts=['relu'],
#                  weight_decay=5e-4, dropout=0.5,
#                  lr=0.01, use_bias=False):

#         super().__init__()

#         self.GNN_layers = []
#         for hid, act, weight_decay in zip(hids, acts, weight_decay):
#             layer = GCNConv(hid, use_bias=bias,
#                                          activation=act,
#                                          kernel_regularizer=regularizers.l2(weight_decay))

#             self.GNN_layers.append(layer)

#         layer = GCNConv(out_features, use_bias=bias)
#         self.GNN_layers.append(layer)

#         self.dropout = Dropout(dropout)
#         self.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
#                       optimizer=Adam(lr), metrics=['accuracy'])

#         self.metrics_fn = SparseCategoricalAccuracy()

#     def call(self, inputs, training=False):
#         x, adj, idx = inputs

#         for layer in self.GNN_layers:
#             x = self.dropout(x, training=training)
#             x = layer([x, adj])

#         return tf.gather(x, idx)
