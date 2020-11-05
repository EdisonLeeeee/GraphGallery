from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery.nn.layers.tensorflow import GraphAttention, Gather
from graphgallery import floatx, intx


class GAT(Model):

    def __init__(self, in_channels,
                 out_channels, hiddens=[16], n_heads=[8],
                 activations=['elu'], dropout=0.6,
                 l2_norm=5e-4,
                 lr=0.01, use_bias=True):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            in_channels: (int): write your description
            out_channels: (int): write your description
            hiddens: (todo): write your description
            n_heads: (int): write your description
            activations: (str): write your description
            dropout: (str): write your description
            l2_norm: (todo): write your description
            lr: (float): write your description
            use_bias: (bool): write your description
        """

        x = Input(batch_shape=[None, in_channels],
                  dtype=floatx(), name='attr_matrix')
        adj = Input(batch_shape=[None, None], dtype=floatx(),
                    sparse=True, name='adj_matrix')
        index = Input(batch_shape=[None], dtype=intx(), name='node_index')

        h = x
        for hidden, n_head, activation in zip(hiddens, n_heads, activations):
            h = GraphAttention(hidden, attn_heads=n_head,
                               reduction='concat',
                               use_bias=use_bias,
                               activation=activation,
                               kernel_regularizer=regularizers.l2(l2_norm),
                               attn_kernel_regularizer=regularizers.l2(
                                   l2_norm),
                               )([h, adj])
            h = Dropout(rate=dropout)(h)

        h = GraphAttention(out_channels, use_bias=use_bias,
                           attn_heads=1, reduction='average')([h, adj])
        h = Gather()([h, index])

        super().__init__(inputs=[x, adj, index], outputs=h)
        self.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                     optimizer=Adam(lr=lr), metrics=['accuracy'])
      # TODO
#     def __repr__(self):
