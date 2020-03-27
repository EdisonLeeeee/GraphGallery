import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import TruncatedNormal

from graphgallery.nn.layers import GraphConvolution
from graphgallery.mapper import FullBatchNodeSequence
from .base import SupervisedModel

from graphgallery.utils import kl_divergence_with_logit, entropy_y_x


class OBVAT(SupervisedModel):
    def __init__(self, adj, features, labels, normalize_rate=-0.5, normalize_features=True, device='CPU:0', seed=None):
    
        super().__init__(adj, features, labels, device=device, seed=seed)
        
        self.normalize_rate = normalize_rate
        self.normalize_features = normalize_features            
        self.preprocess(adj, features)
        self.do_before_train = self.extra_train

    def preprocess(self, adj, features):
        
        if self.normalize_rate is not None:
            adj = self._normalize_adj(adj, self.normalize_rate)        
            
        if self.normalize_features:
            features = self._normalize_features(features)
            
        with self.device:
            self.features, self.adj = self._to_tensor([features, adj])


    def build(self, hidden_layers=[32], activations=['relu'], dropout=0.5, learning_rate=0.01, l2_norm=5e-4, p1=1.4, p2=0.7):

        with self.device:

            x = Input(batch_shape=[self.n_nodes, self.n_features], dtype=tf.float32, name='features')
            adj = Input(batch_shape=[self.n_nodes, self.n_nodes], dtype=tf.float32, sparse=True, name='adj_matrix')
            index = Input(batch_shape=[None],  dtype=tf.int32, name='index')

            self.GCN_layers = [GraphConvolution(hidden_layers[0], activation=activations[0], 
                                                kernel_regularizer=regularizers.l2(l2_norm)),
                               GraphConvolution(self.n_classes)]
            self.dropout_layer = Dropout(rate=dropout)

            logit = self.propagation(x, adj)
            logit = tf.ensure_shape(logit, (self.n_nodes, self.n_classes))
            output = tf.gather(logit, index)
            output = Softmax()(output)
            model = Model(inputs=[x, adj, index], outputs=output)
            model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])

            self.r_vadv = tf.Variable(TruncatedNormal(stddev=0.01)(shape=[self.n_nodes, self.n_features]), name="r_vadv")
            entropy_loss = entropy_y_x(logit)
            vat_loss = self.virtual_adversarial_loss(x, adj, logit)
            model.add_loss(p1 * vat_loss + p2 * entropy_loss)

            self.model = model
            self.adv_optimizer = Adam(lr=learning_rate/10)
            self.built = True

    def virtual_adversarial_loss(self, x, adj, logit):

        adv_x = x + self.r_vadv
        logit_p = tf.stop_gradient(logit)
        logit_m = self.propagation(adv_x, adj)
        loss = kl_divergence_with_logit(logit_p, logit_m)
        return loss
    
    def propagation(self, x, adj):
        h = x
        for layer in self.GCN_layers:
            h = self.dropout_layer(h)
            h = layer([h, adj])
        return h

    @tf.function
    def extra_train(self, epochs=10):
        r_vadv = self.r_vadv
        optimizer = self.adv_optimizer
        x, adj = self.features, self.adj
        r_vadv.assign(TruncatedNormal(stddev=0.01)(shape=tf.shape(r_vadv)))
        with self.device:
            for _ in range(epochs):
                with tf.GradientTape() as tape:
                    rnorm = tf.nn.l2_loss(r_vadv)
                    logit = self.propagation(x, adj)
                    vloss = self.virtual_adversarial_loss(x, adj, logit)
                    loss = rnorm - vloss
                gradient = tape.gradient(loss, r_vadv)
                optimizer.apply_gradients(zip([gradient], [r_vadv]))

    def predict(self, index):
        super().predict(index)
        index = self._check_and_convert(index)
        
        with self.device:
            index = self._to_tensor(index)
            logit = self.model.predict_on_batch([self.features, self.adj, index])

        return logit.numpy()                                                                                              
            
    def train_sequence(self, index):
        index = self._check_and_convert(index)
        labels = self.labels[index]
           
        with self.device:
            sequence = FullBatchNodeSequence([self.features, self.adj, index], labels)
            
        return sequence    