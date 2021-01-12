from tensorflow.keras import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery.nn.layers.tensorflow import GraphAttention
from graphgallery import floatx, intx
from graphgallery.nn.models import TFKeras


class GAT(TFKeras):

    def __init__(self, in_channels,
                 out_channels, hids=[16], num_heads=[8],
                 acts=['elu'], dropout=0.6,
                 weight_decay=5e-4,
                 lr=0.01, use_bias=True):

        x = Input(batch_shape=[None, in_channels],
                  dtype=floatx(), name='node_attr')
        adj = Input(batch_shape=[None, None], dtype=floatx(),
                    sparse=True, name='adj_matrix')

        h = x
        for hid, num_head, act in zip(hids, num_heads, acts):
            h = GraphAttention(hid, attn_heads=num_head,
                               reduction='concat',
                               use_bias=use_bias,
                               activation=act,
                               kernel_regularizer=regularizers.l2(weight_decay),
                               attn_kernel_regularizer=regularizers.l2(
                                   weight_decay),
                               )([h, adj])
            h = Dropout(rate=dropout)(h)

        h = GraphAttention(out_channels, use_bias=use_bias,
                           attn_heads=1, reduction='average')([h, adj])

        super().__init__(inputs=[x, adj], outputs=h)
        self.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                     optimizer=Adam(lr=lr), metrics=['accuracy'])
      # TODO
#     def __repr__(self):
