from tensorflow.keras import Input
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy


from graphgallery.nn.layers.tensorflow import GraphConvolution
from graphgallery import floatx, intx
from graphgallery.nn.models import TFKeras


class MLP(TFKeras):

    def __init__(self, in_channels, out_channels,
                 hids=[16],
                 acts=['relu'],
                 dropout=0.5,
                 weight_decay=5e-4,
                 lr=0.01, use_bias=False,
                 experimental_run_tf_function=True):

        x = Input(batch_shape=[None, in_channels],
                  dtype=floatx(), name='node_attr')

        h = x
        for hid, act in zip(hids, acts):
            h = Dense(hid, use_bias=use_bias,
                                 activation=act,
                                 kernel_regularizer=regularizers.l2(weight_decay))(h)

            h = Dropout(rate=dropout)(h)

        h = Dense(out_channels, use_bias=use_bias)(h)

        super().__init__(inputs=x, outputs=h)
        self.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                     optimizer=Adam(lr=lr), metrics=['accuracy'],
                     experimental_run_tf_function=experimental_run_tf_function)

