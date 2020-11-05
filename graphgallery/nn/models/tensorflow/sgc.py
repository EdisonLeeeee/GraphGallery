from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery import floatx, intx

                
class SGC(Model):

    def __init__(self, in_channels,
                 out_channels, hiddens=[], 
                 activations=[], 
                 dropout=0.5,
                 l2_norm=5e-5, 
                 lr=0.2, use_bias=False):
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

        if len(hiddens) != len(activations):
            raise RuntimeError(f"Arguments 'hiddens' and 'activations' should have the same length."
                               " Or you can set both of them to `[]`.")

        x = Input(batch_shape=[None, in_channels],
                  dtype=floatx(), name='attr_matrix')
        
        h = x
        for hidden, activation in zip(hiddens, activations):
            h = Dropout(dropout)(h)
            h = Dense(hidden, activation=activation, use_bias=use_bias,
                      kernel_regularizer=regularizers.l2(l2_norm))(h)
            
        h = Dropout(dropout)(h)
        output = Dense(out_channels, activation=None, use_bias=use_bias,
                  kernel_regularizer=regularizers.l2(l2_norm))(h)

        super().__init__(inputs=x, outputs=output)
        self.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=Adam(lr=lr), metrics=['accuracy'])        

