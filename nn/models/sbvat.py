import numpy as np
import tensorflow as tf

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.activations import softmax
from tensorflow.keras.metrics import SparseCategoricalAccuracy

from graphgallery.nn.layers import GraphConvolution
from graphgallery.mapper import NodeSampleSequence
from graphgallery.utils import find_4o_nbrs, get_normalized_vector, kl_divergence_with_logit, entropy_y_x
from .base import SupervisedModel


class SBVAT(SupervisedModel):
    def __init__(self, adj, features, labels, n_sample=100, 
                 normalize_rate=-0.5, normalize_features=True, device='CPU:0', seed=None):
    
        super().__init__(adj, features, labels, device=device, seed=seed)
        
        self.normalize_rate = normalize_rate
        self.normalize_features = normalize_features            
        self.preprocess(adj, features)
        self.n_sample = n_sample

    def preprocess(self, adj, features):
        
        if self.normalize_rate is not None:
            adj = self._normalize_adj(adj, self.normalize_rate)        
            
        if self.normalize_features:
            features = self._normalize_features(features)
            
        self.neighbors = list(find_4o_nbrs(adj.indices, adj.indptr, np.arange(self.n_nodes)))

        with self.device:
            self.features, self.adj = self._to_tensor([features, adj])


    def build(self, hidden_layers=[32], activations=['relu'], dropout=0.5, 
              learning_rate=0.01, l2_norm=5e-4, p1=1., p2=1., 
              n_power_iterations=1, epsilon=0.03, xi=1e-6):
        
        with self.device:
            
            x = Input(batch_shape=[self.n_nodes, self.n_features], dtype=tf.float32, name='features')
            adj = Input(batch_shape=[self.n_nodes, self.n_nodes], dtype=tf.float32, sparse=True, name='adj_matrix')
            index = Input(batch_shape=[None],  dtype=tf.int32, name='index')

            self.GCN_layers = [GraphConvolution(hidden_layers[0], 
                                                activation=activations[0], 
                                                kernel_regularizer=regularizers.l2(l2_norm)),
                               GraphConvolution(self.n_classes)]
            self.dropout_layer = Dropout(dropout)
            
            logit = self.propagation(x, adj)
            output = tf.gather(logit, index)
            output = Softmax()(output)
            model = Model(inputs=[x, adj, index], outputs=output)
    
            self.model = model
            self.train_metric = SparseCategoricalAccuracy()
            self.test_metric = SparseCategoricalAccuracy()
            self.optimizer = Adam(lr=learning_rate)
            self.built = True
            
        self.p1 = p1 # Alpha
        self.p2 = p2 # Beta
        self.xi = xi # Small constant for finite difference
        self.epsilon = epsilon # Norm length for (virtual) adversarial training
        self.n_power_iterations = n_power_iterations #  Number of power iterations
            
    def propagation(self, x, adj, training=True):
        h = x
        for layer in self.GCN_layers:
            h = self.dropout_layer(h, training=training)
            h = layer([h, adj])
        return h
    
    @tf.function
    def do_train_forward(self, sequence):
        
        with self.device:
            self.train_metric.reset_states()
            
            for inputs, labels in sequence:
                x, adj, index, adv_mask = inputs
                with tf.GradientTape() as tape:
                    logit = self.propagation(x, adj)
                    output = tf.gather(logit, index)
                    output = softmax(output)

                    loss = tf.reduce_mean(sparse_categorical_crossentropy(labels, output))
                    entropy_loss = entropy_y_x(logit)
                    vat_loss = self.virtual_adversarial_loss(x, adj, logit=logit, adv_mask=adv_mask)
                    loss += self.p1 * vat_loss + self.p2 * entropy_loss
            
                    self.train_metric.update_state(labels, output)

                trainable_variables = self.model.trainable_variables
                gradients = tape.gradient(loss, trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss, self.train_metric.result()
            
    @tf.function
    def do_test_forward(self, sequence):
            
        with self.device:
            self.test_metric.reset_states()
            
            for inputs, labels in sequence:
                x, adj, index, _ = inputs
                logit = self.propagation(x, adj, training=False)
                output = tf.gather(logit, index)
                output = softmax(output)
                loss = tf.reduce_mean(sparse_categorical_crossentropy(labels, output))
                self.test_metric.update_state(labels, output)
            
        return loss, self.test_metric.result()
        
        
    def do_forward(self, sequence, training=True):
        if training:
            loss, accuracy = self.do_train_forward(sequence)
        else:
            loss, accuracy = self.do_test_forward(sequence)
            
        return loss.numpy(), accuracy.numpy()

    
    def virtual_adversarial_loss(self, x, adj, logit, adv_mask):
        d = tf.random.normal(shape=tf.shape(x))
        
        for _ in range(self.n_power_iterations):
            d = get_normalized_vector(d) * self.xi
            logit_p = logit
            with tf.GradientTape() as tape:
                tape.watch(d)
                logit_m = self.propagation(x + d, adj)
                dist = kl_divergence_with_logit(logit_p, logit_m, adv_mask)
            grad = tape.gradient(dist, d)
            d = tf.stop_gradient(grad)

        r_vadv = get_normalized_vector(d) * self.epsilon
        logit_p = tf.stop_gradient(logit)
        logit_m = self.propagation(x + r_vadv, adj)
        loss = kl_divergence_with_logit(logit_p, logit_m, adv_mask)
        return tf.identity(loss)    

    
    def train_sequence(self, index):
        index = self._check_and_convert(index)
        labels = self.labels[index]
           
        with self.device:
            sequence = NodeSampleSequence([self.features, self.adj, index], labels,
                                          neighbors=self.neighbors,
                                          n_sample=self.n_sample)
            
        return sequence    
    
    def test_sequence(self, index):
        index = self._check_and_convert(index)
        labels = self.labels[index]
           
        with self.device:
            sequence = NodeSampleSequence([self.features, self.adj, index], labels,
                                          neighbors=self.neighbors,
                                          n_sample=self.n_sample,
                                          resample=False)
            
        return sequence        
    
    def predict(self, index):
        super().predict(index)
        index = self._check_and_convert(index)
        
        with self.device:
            sequence = NodeSampleSequence([self.features, self.adj, index], None,
                                          neighbors=self.neighbors,
                                          n_sample=self.n_sample,
                                          resample=False)
            for inputs, _ in sequence:
                x, adj, index, adv_mask = inputs
                output = self.propagation(x, adj, training=False)
                logit = softmax(tf.gather(output, index))
                
        return logit.numpy()      