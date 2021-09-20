from tensorflow.keras import Input
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy


from graphgallery import floatx
from graphgallery.nn.models.tf_keras import TFKeras


class MLP(TFKeras):

    def __init__(self, in_features, out_features,
                 hids=[16],
                 acts=['relu'],
                 dropout=0.5,
                 weight_decay=5e-4,
                 lr=0.01, bias=False):

        if len(hids) != len(acts):
            raise RuntimeError(f"Arguments 'hids' and 'acts' should have the same length."
                               " Or you can set both of them to `[]`.")

        x = Input(batch_shape=[None, in_features],
                  dtype=floatx(), name='node_attr')

        h = x
        for hid, act in zip(hids, acts):
            h = Dropout(rate=dropout)(h)
            h = Dense(hid, use_bias=bias,
                      activation=act,
                      kernel_regularizer=regularizers.l2(weight_decay))(h)

        h = Dropout(rate=dropout)(h)
        h = Dense(out_features, use_bias=bias,
                  kernel_regularizer=regularizers.l2(weight_decay))(h)

        super().__init__(inputs=x, outputs=h)
        self.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                     optimizer=Adam(lr=lr), metrics=['accuracy'])
