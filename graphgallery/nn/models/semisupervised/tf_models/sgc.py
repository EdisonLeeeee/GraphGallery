from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery import floatx, intx

                
class SGC(Model):

    def __init__(self, in_channels,
                 out_channels, hiddens=[], activations=[],
                 l2_norms=[5e-5], dropout=0.5, 
                 lr=0.2, use_bias=False):

        if len(hiddens) != len(activations):
            raise RuntimeError(f"Arguments 'hiddens' and 'activations' should have the same length."
                               " Or you can set both of them to `[]`.")

        x = Input(batch_shape=[None, in_channels],
                  dtype=floatx(), name='attr_matrix')
        
        h = x
        for hidden, activation, l2_norm in zip(hiddens, activations, l2_norms):
            h = Dropout(dropout)(h)
            h = Dense(hidden, activation=activation, use_bias=use_bias,
                      kernel_regularizer=regularizers.l2(l2_norm))(h)
            
        h = Dropout(dropout)(h)
        output = Dense(out_channels, activation=None, use_bias=use_bias,
                  kernel_regularizer=regularizers.l2(l2_norms[-1]))(h)

        super().__init__(inputs=x, outputs=output)
        self.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=Adam(lr=lr), metrics=['accuracy'])        

