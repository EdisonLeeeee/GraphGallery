import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

from graphgallery.nn.layers.tf_layers import GraphConvolution

class GCN(Model):

    def __init__(self, hiddens,
                 out_channels, activations=['relu'],
                 l2_norms=[5e-4], dropout=0.5, 
                 lr=0.01, use_bias=False):

        super().__init__()

        self.GNN_layers = []
        for hid, activation, l2_norm in zip(hiddens, activations, l2_norms):
            layer = GraphConvolution(hid, use_bias=use_bias,
                                         activation=activation,
                                         kernel_regularizer=regularizers.l2(l2_norm))
            
            self.GNN_layers.append(layer)

        layer = GraphConvolution(out_channels, use_bias=use_bias)
        self.GNN_layers.append(layer)
        
        self.dropout = Dropout(dropout)
        self.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=Adam(lr=lr), metrics=['accuracy'])

        self.metrics_fn = SparseCategoricalAccuracy()        

    def call(self, inputs, training=False):
        x, adj, idx = inputs

        for layer in self.GNN_layers:
            x = self.dropout(x, training=training)
            x = layer([x, adj])
            
        return tf.gather(x, idx)
    