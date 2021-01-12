from tensorflow.keras import Input
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery.nn.models import TFKeras
from graphgallery import floatx


class SGC(TFKeras):

    def __init__(self, in_channels,
                 out_channels, hids=[],
                 acts=[],
                 dropout=0.5,
                 weight_decay=5e-5,
                 lr=0.2, use_bias=False):

        if len(hids) != len(acts):
            raise RuntimeError(f"Arguments 'hids' and 'acts' should have the same length."
                               " Or you can set both of them to `[]`.")

        x = Input(batch_shape=[None, in_channels],
                  dtype=floatx(), name='node_attr')

        h = x
        for hid, act in zip(hids, acts):
            h = Dropout(dropout)(h)
            h = Dense(hid, activation=act, use_bias=use_bias,
                      kernel_regularizer=regularizers.l2(weight_decay))(h)

        h = Dropout(dropout)(h)
        output = Dense(out_channels, activation=None, use_bias=use_bias,
                       kernel_regularizer=regularizers.l2(weight_decay))(h)

        super().__init__(inputs=x, outputs=output)
        self.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                     optimizer=Adam(lr=lr), metrics=['accuracy'])
