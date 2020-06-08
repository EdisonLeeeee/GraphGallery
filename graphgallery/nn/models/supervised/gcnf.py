import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from graphgallery.nn.layers import GraphConvFeature
from graphgallery.sequence import FullBatchNodeSequence
from graphgallery.nn.models import SupervisedModel


class GCNF(SupervisedModel):


    def __init__(self, adj, x, labels, normalize_rate=-0.5, is_normalize_x=True,
                 device='CPU:0', seed=None, name=None, **kwargs):

        super().__init__(adj, x, labels, device=device, seed=seed, name=name, **kwargs)

        self.normalize_rate = normalize_rate
        self.is_normalize_x = is_normalize_x
        self.preprocess(adj, x)

    def preprocess(self, adj, x):
        adj, x = super().preprocess(adj, x)

        if self.normalize_rate is not None:
            adj = self.normalize_adj(adj, self.normalize_rate)

        if self.is_normalize_x:
            x = self.normalize_x(x)

        with tf.device(self.device):
            self.tf_x, self.tf_adj = self.to_tensor([x, adj])

    def build(self, hiddens=[16], activations=['relu'], dropout=0.5,
              lr=0.01, l2_norm=5e-4, use_bias=False, ensure_shape=True):
        
        assert len(hiddens) == len(activations)
        
        with tf.device(self.device):

            x = Input(batch_shape=[None, self.n_features], dtype=self.floatx, name='features')
            adj = Input(batch_shape=[None, None], dtype=self.floatx, sparse=True, name='adj_matrix')
            index = Input(batch_shape=[None], dtype=self.intx, name='index')

            h = x
            for hid, activation in zip(hiddens, activations):
                h = GraphConvFeature(hid, use_bias=use_bias,
                                     activation=activation,
                                     concat=True,
                                     kernel_regularizer=regularizers.l2(l2_norm))([h, adj])

                h = Dropout(rate=dropout)(h)
                

            h = GraphConvFeature(self.n_classes, use_bias=use_bias)([h, adj])
            if ensure_shape:
                h = tf.ensure_shape(h, [self.n_nodes, self.n_classes])
            h = tf.gather(h, index)
            output = Softmax()(h)

            model = Model(inputs=[x, adj, index], outputs=output)
            model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])

            self.set_model(model)
            self.built = True

    def train_sequence(self, index):
        index = self.to_int(index)
        labels = self.labels[index]
        with tf.device(self.device):
            sequence = FullBatchNodeSequence([self.tf_x, self.tf_adj, index], labels)
        return sequence

    def predict(self, index):
        super().predict(index)
        index = self.to_int(index)
        with tf.device(self.device):
            index = self.to_tensor(index)
            logit = self.model.predict_on_batch([self.tf_x, self.tf_adj, index])

        if tf.is_tensor(logit):
            logit = logit.numpy()
        return logit
