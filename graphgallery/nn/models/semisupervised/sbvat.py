import numpy as np
import tensorflow as tf

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.activations import softmax
from tensorflow.keras.metrics import SparseCategoricalAccuracy

from graphgallery.nn.layers import GraphConvolution, Gather
from graphgallery.nn.models import SemiSupervisedModel
from graphgallery.sequence import NodeSampleSequence
from graphgallery.utils.sample import find_4o_nbrs
from graphgallery.utils.bvat_utils import get_normalized_vector, kl_divergence_with_logit, entropy_y_x
from graphgallery.utils.shape import set_equal_in_length
from graphgallery import astensors, asintarr, normalize_x, normalize_adj, Bunch


class SBVAT(SemiSupervisedModel):
    """
        Implementation of optimization-based Batch Virtual Adversarial Training  Graph Convolutional Networks (OBVAT). 
        `Batch Virtual Adversarial Training for Graph Convolutional Networks <https://arxiv.org/abs/1902.09192>`
        Tensorflow 1.x implementation: <https://github.com/thudzj/BVAT>

        Arguments:
        ----------
            adj: shape (N, N), Scipy sparse matrix if  `is_adj_sparse=True`, 
                Numpy array-like (or matrix) if `is_adj_sparse=False`.
                The input `symmetric` adjacency matrix, where `N` is the number 
                of nodes in graph.
            x: shape (N, F), Scipy sparse matrix if `is_x_sparse=True`, 
                Numpy array-like (or matrix) if `is_x_sparse=False`.
                The input node feature matrix, where `F` is the dimension of features.
            labels: Numpy array-like with shape (N,)
                The ground-truth labels for all nodes in graph.
            n_samples (Positive integer, optional): 
                The number of sampled subset nodes in the graph where the shortest path 
                length between them is at least 4. (default :obj: `50`)
            norm_adj (Float scalar, optional): 
                The normalize rate for adjacency matrix `adj`. (default: :obj:`-0.5`, 
                i.e., math:: \hat{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}}) 
            norm_x (String, optional): 
                How to normalize the node feature matrix. See `graphgallery.normalize_x`
                (default :obj: `None`)
            device (String, optional): 
                The device where the model is running on. You can specified `CPU` or `GPU` 
                for the model. (default: :str: `CPU:0`, i.e., running on the 0-th `CPU`)
            seed (Positive integer, optional): 
                Used in combination with `tf.random.set_seed` & `np.random.seed` & `random.seed`  
                to create a reproducible sequence of tensors across multiple calls. 
                (default :obj: `None`, i.e., using random seed)
            name (String, optional): 
                Specified name for the model. (default: :str: `class.__name__`)

    """

    def __init__(self, adj, x, labels, n_samples=100,
                 norm_adj=-0.5, norm_x=None, device='CPU:0', seed=None, name=None, **kwargs):

        super().__init__(adj, x, labels, device=device, seed=seed, name=name, **kwargs)

        self.norm_adj = norm_adj
        self.norm_x = norm_x
        self.n_samples = n_samples
        self.preprocess(adj, x)

    def preprocess(self, adj, x):
        super().preprocess(adj, x)
        adj, x = self.adj, self.x

        if self.norm_adj:
            adj = normalize_adj(adj, self.norm_adj)

        if self.norm_x:
            x = normalize_x(x, norm=self.norm_x)

        self.neighbors = list(find_4o_nbrs(adj.indices, adj.indptr, np.arange(self.n_nodes)))

        with tf.device(self.device):
            self.x_norm, self.adj_norm = astensors([x, adj])

    def build(self, hiddens=[16], activations=['relu'], dropouts=[0.5],
              lr=0.01, l2_norms=[5e-4], p1=1., p2=1., use_bias=False,
              n_power_iterations=1, epsilon=0.03, xi=1e-6):

        ############# Record paras ###########
        local_paras = locals()
        local_paras.pop('self')
        paras = Bunch(**local_paras)
        hiddens, activations, dropouts, l2_norms = set_equal_in_length(hiddens, activations, dropouts, l2_norms)
        paras.update(Bunch(hiddens=hiddens, activations=activations, dropouts=dropouts, l2_norms=l2_norms))
        # update all parameters
        self.paras.update(paras)
        self.model_paras.update(paras)
        ######################################

        with tf.device(self.device):

            x = Input(batch_shape=[self.n_nodes, self.n_features], dtype=self.floatx, name='features')
            adj = Input(batch_shape=[self.n_nodes, self.n_nodes], dtype=self.floatx, sparse=True, name='adj_matrix')
            index = Input(batch_shape=[None],  dtype=self.intx, name='index')

            GCN_layers = []
            dropout_layers = []
            for hid, activation, dropout, l2_norm in zip(hiddens, activations, dropouts, l2_norms):
                GCN_layers.append(GraphConvolution(hid, activation=activation, use_bias=use_bias,
                                                   kernel_regularizer=regularizers.l2(l2_norm)))
                dropout_layers.append(Dropout(rate=dropout))

            GCN_layers.append(GraphConvolution(self.n_classes, use_bias=use_bias))
            self.GCN_layers = GCN_layers
            self.dropout_layers = dropout_layers

            logit = self.propagation(x, adj)
            output = Gather()([logit, index])
            model = Model(inputs=[x, adj, index], outputs=output)

            self.model = model
            self.train_metric = SparseCategoricalAccuracy()
            self.test_metric = SparseCategoricalAccuracy()
            self.optimizer = Adam(lr=lr)

        self.p1 = p1  # Alpha
        self.p2 = p2  # Beta
        self.xi = xi  # Small constant for finite difference
        self.epsilon = epsilon  # Norm length for (virtual) adversarial training
        self.n_power_iterations = n_power_iterations  # Number of power iterations

    def propagation(self, x, adj, training=True):
        h = x
        for dropout_layer, GCN_layer in zip(self.dropout_layers, self.GCN_layers[:-1]):
            h = dropout_layer(h, training=training)
            h = GCN_layer([h, adj])
        h = dropout_layer(h, training=training)
        h = self.GCN_layers[-1]([h, adj])
        return h

    @tf.function
    def train_step(self, sequence):

        with tf.device(self.device):
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
    def test_step(self, sequence):

        with tf.device(self.device):
            self.test_metric.reset_states()

            for inputs, labels in sequence:
                x, adj, index, _ = inputs
                logit = self.propagation(x, adj, training=False)
                output = tf.gather(logit, index)
                output = softmax(output)
                loss = tf.reduce_mean(sparse_categorical_crossentropy(labels, output))
                self.test_metric.update_state(labels, output)

        return loss, self.test_metric.result()

    def virtual_adversarial_loss(self, x, adj, logit, adv_mask):
        d = tf.random.normal(shape=tf.shape(x), dtype=self.floatx)

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
        index = asintarr(index)
        labels = self.labels[index]

        with tf.device(self.device):
            sequence = NodeSampleSequence([self.x_norm, self.adj_norm, index], labels,
                                          neighbors=self.neighbors,
                                          n_samples=self.n_samples)

        return sequence

    def test_sequence(self, index):
        index = asintarr(index)
        labels = self.labels[index]

        with tf.device(self.device):
            sequence = NodeSampleSequence([self.x_norm, self.adj_norm, index], labels,
                                          neighbors=self.neighbors,
                                          n_samples=self.n_samples,
                                          resample=False)

        return sequence

    def predict(self, index):
        super().predict(index)
        index = asintarr(index)

        with tf.device(self.device):
            sequence = NodeSampleSequence([self.x_norm, self.adj_norm, index], None,
                                          neighbors=self.neighbors,
                                          n_samples=self.n_samples,
                                          resample=False)
            for inputs, _ in sequence:
                x, adj, index, adv_mask = inputs
                output = self.propagation(x, adj, training=False)
                logit = softmax(tf.gather(output, index))

        if tf.is_tensor(logit):
            logit = logit.numpy()
        return logit
