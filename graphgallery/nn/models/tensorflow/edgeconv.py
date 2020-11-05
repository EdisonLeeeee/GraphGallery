from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery.nn.layers.tensorflow import GraphEdgeConvolution, Gather
from graphgallery import floatx, intx


class EdgeGCN(Model):

    def __init__(self, in_channels, out_channels,
                 hiddens=[16], activations=['relu'], dropout=0.5,
                 l2_norm=5e-4, lr=0.01, use_bias=False):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            in_channels: (int): write your description
            out_channels: (int): write your description
            hiddens: (todo): write your description
            activations: (str): write your description
            dropout: (str): write your description
            l2_norm: (todo): write your description
            lr: (float): write your description
            use_bias: (bool): write your description
        """

        _intx = intx()
        _floatx = floatx()
        x = Input(batch_shape=[None, in_channels],
                  dtype=_floatx, name='attr_matrix')
        edge_index = Input(batch_shape=[None, 2], dtype=_intx,
                           name='edge_index')
        edge_weight = Input(batch_shape=[None], dtype=_floatx,
                            name='edge_weight')
        index = Input(batch_shape=[None],
                      dtype=_intx, name='node_index')

        h = x
        for hidden, activation in zip(hiddens, activations):
            h = GraphEdgeConvolution(hidden, use_bias=use_bias,
                                     activation=activation,
                                     kernel_regularizer=regularizers.l2(l2_norm))([h, edge_index, edge_weight])

            h = Dropout(rate=dropout)(h)

        h = GraphEdgeConvolution(out_channels, use_bias=use_bias)(
            [h, edge_index, edge_weight])
        output = Gather()([h, index])

        super().__init__(inputs=[x, edge_index, edge_weight, index], outputs=output)
        self.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                     optimizer=Adam(lr=lr), metrics=['accuracy'])
