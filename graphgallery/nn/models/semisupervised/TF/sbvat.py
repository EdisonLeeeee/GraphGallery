import numpy as np
import tensorflow as tf

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.activations import softmax
from tensorflow.keras.metrics import SparseCategoricalAccuracy

from graphgallery.nn.layers import GraphConvolution, Gather
from graphgallery.nn.models import SemiSupervisedModel
from graphgallery.sequence import SBVATSampleSequence
from graphgallery.utils.sample import find_4o_nbrs
from graphgallery.utils.bvat_utils import get_normalized_vector, kl_divergence_with_logit, entropy_y_x
from graphgallery.utils.decorators import EqualVarLength
from graphgallery import transformers as T


class SBVAT(SemiSupervisedModel):
    """
        Implementation of sample-based Batch Virtual Adversarial Training
        Graph Convolutional Networks (SBVAT).
        `Batch Virtual Adversarial Training for Graph Convolutional Networks
        <https://arxiv.org/abs/1902.09192>`
        Tensorflow 1.x implementation: <https://github.com/thudzj/BVAT>


    """

    def __init__(self, *graph, n_samples=50,
                 adj_transformer="normalize_adj", attr_transformer=None,
                 device='cpu:0', seed=None, name=None, **kwargs):
        """Creat a sample-based Batch Virtual Adversarial Training
        Graph Convolutional Networks (SBVAT) model.

         This can be instantiated in several ways:

            model = SBVAT(graph)
                with a `graphgallery.data.Graph` instance representing
                A sparse, attributed, labeled graph.

            model = SBVAT(adj_matrix, attr_matrix, labels)
                where `adj_matrix` is a 2D Scipy sparse matrix denoting the graph,
                 `attr_matrix` is a 2D Numpy array-like matrix denoting the node
                 attributes, `labels` is a 1D Numpy array denoting the node labels.


        Parameters:
        ----------
        graph: An instance of `graphgallery.data.Graph` or a tuple (list) of inputs.
            A sparse, attributed, labeled graph.
        n_samples (Positive integer, optional):
            The number of sampled subset nodes in the graph where the length of the
            shortest path between them is at least `4`. (default :obj: `50`)
        adj_transformer: string, `transformer`, or None. optional
            How to transform the adjacency matrix. See `graphgallery.transformers`
            (default: :obj:`'normalize_adj'` with normalize rate `-0.5`.
            i.e., math:: \hat{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}})
        attr_transformer: string, transformer, or None. optional
            How to transform the node attribute matrix. See `graphgallery.transformers`
            (default :obj: `None`)
        device: string. optional
            The device where the model is running on. You can specified `CPU` or `GPU`
            for the model. (default: :str: `CPU:0`, i.e., running on the 0-th `CPU`)
        seed: interger scalar. optional
            Used in combination with `tf.random.set_seed` & `np.random.seed`
            & `random.seed` to create a reproducible sequence of tensors across
            multiple calls. (default :obj: `None`, i.e., using random seed)
        name: string. optional
            Specified name for the model. (default: :str: `class.__name__`)
        kwargs: other customed keyword Parameters.
        """
        super().__init__(*graph, device=device, seed=seed, name=name, **kwargs)

        self.adj_transformer = T.get(adj_transformer)
        self.attr_transformer = T.get(attr_transformer)
        self.n_samples = n_samples
        self.process()

    def process_step(self):
        graph = self.graph
        adj_matrix = self.adj_transformer(graph.adj_matrix)
        attr_matrix = self.attr_transformer(graph.attr_matrix)
        self.neighbors = find_4o_nbrs(adj_matrix)

        self.feature_inputs, self.structure_inputs = T.astensors(
            attr_matrix, adj_matrix, device=self.device)

    # use decorator to make sure all list arguments have the same length
    @EqualVarLength()
    def build(self, hiddens=[16], activations=['relu'], dropouts=[0.5],
              lr=0.01, l2_norms=[5e-4], use_bias=False, p1=1., p2=1.,
              n_power_iterations=1, epsilon=0.03, xi=1e-6):

        with tf.device(self.device):

            x = Input(batch_shape=[None, self.graph.n_attrs],
                      dtype=self.floatx, name='attr_matrix')
            adj = Input(batch_shape=[None, None],
                        dtype=self.floatx, sparse=True, name='adj_matrix')
            index = Input(batch_shape=[None],
                          dtype=self.intx, name='node_index')

            GCN_layers = []
            dropout_layers = []
            for hid, activation, dropout, l2_norm in zip(hiddens, activations, dropouts, l2_norms):
                GCN_layers.append(GraphConvolution(hid, activation=activation, use_bias=use_bias,
                                                   kernel_regularizer=regularizers.l2(l2_norm)))
                dropout_layers.append(Dropout(rate=dropout))

            GCN_layers.append(GraphConvolution(
                self.graph.n_classes, use_bias=use_bias))
            self.GCN_layers = GCN_layers
            self.dropout_layers = dropout_layers

            logit = self.forward(x, adj)
            output = Gather()([logit, index])
            model = Model(inputs=[x, adj, index], outputs=output)

            self.model = model
            self.train_metric = SparseCategoricalAccuracy()
            self.test_metric = SparseCategoricalAccuracy()
            self.loss_fn = SparseCategoricalCrossentropy(from_logits=True)
            self.optimizer = Adam(lr=lr)

        self.p1 = p1  # Alpha
        self.p2 = p2  # Beta
        self.xi = xi  # Small constant for finite difference
        # Norm length for (virtual) adversarial training
        self.epsilon = epsilon
        self.n_power_iterations = n_power_iterations  # Number of power iterations

    def forward(self, x, adj, training=True):
        h = x
        for dropout_layer, GCN_layer in zip(self.dropout_layers, self.GCN_layers[:-1]):
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
                    logit = self.forward(x, adj)
                    output = tf.gather(logit, index)
                    loss = self.loss_fn(labels, output)
                    entropy_loss = entropy_y_x(logit)
                    vat_loss = self.virtual_adversarial_loss(
                        x, adj, logit=logit, adv_mask=adv_mask)
                    loss += self.p1 * vat_loss + self.p2 * entropy_loss

                    self.train_metric.update_state(labels, output)

                trainable_variables = self.model.trainable_variables
                gradients = tape.gradient(loss, trainable_variables)
                self.optimizer.apply_gradients(
                    zip(gradients, trainable_variables))

            return loss, self.train_metric.result()

    @tf.function
    def test_step(self, sequence):

        with tf.device(self.device):
            self.test_metric.reset_states()

            for inputs, labels in sequence:
                x, adj, index, _ = inputs
                logit = self.forward(x, adj, training=False)
                output = tf.gather(logit, index)
                loss = self.loss_fn(labels, output)
                self.test_metric.update_state(labels, output)

            return loss, self.test_metric.result()

    def virtual_adversarial_loss(self, x, adj, logit, adv_mask):
        d = tf.random.normal(shape=tf.shape(x), dtype=self.floatx)

        for _ in range(self.n_power_iterations):
            d = get_normalized_vector(d) * self.xi
            logit_p = logit
            with tf.GradientTape() as tape:
                tape.watch(d)
                logit_m = self.forward(x + d, adj)
                dist = kl_divergence_with_logit(logit_p, logit_m, adv_mask)
            grad = tape.gradient(dist, d)
            d = tf.stop_gradient(grad)

        r_vadv = get_normalized_vector(d) * self.epsilon
        logit_p = tf.stop_gradient(logit)
        logit_m = self.forward(x + r_vadv, adj)
        loss = kl_divergence_with_logit(logit_p, logit_m, adv_mask)
        return tf.identity(loss)

    def train_sequence(self, index):
        index = T.asintarr(index)
        labels = self.graph.labels[index]

        sequence = SBVATSampleSequence([self.feature_inputs, self.structure_inputs,
                                        index], labels,
                                        neighbors=self.neighbors,
                                        n_samples=self.n_samples, device=self.device)

        return sequence

    def test_sequence(self, index):
        index = T.asintarr(index)
        labels = self.graph.labels[index]

        sequence = SBVATSampleSequence([self.feature_inputs, self.structure_inputs,
                                        index], labels,
                                        neighbors=self.neighbors,
                                        n_samples=self.n_samples,
                                        resample=False, device=self.device)

        return sequence

    def predict_step(self, sequence):
        with tf.device(self.device):
            for inputs, _ in sequence:
                x, adj, index, adv_mask = inputs
                output = self.forward(x, adj, training=False)
                logit = tf.gather(output, index)

        if tf.is_tensor(logit):
            logit = logit.numpy()
        return logit
