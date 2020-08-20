import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery.nn.layers import GraphConvolution, Gather
from graphgallery.nn.models import SemiSupervisedModel
from graphgallery.sequence import FullBatchNodeSequence
from graphgallery.utils.bvat_utils import kl_divergence_with_logit, entropy_y_x
from graphgallery.utils.shape import set_equal_in_length
from graphgallery import astensors, asintarr, normalize_x, normalize_adj, Bunch


class OBVAT(SemiSupervisedModel):
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

    def __init__(self, adj, x, labels, norm_adj=-0.5, norm_x=None,
                 device='CPU:0', seed=None, name=None, **kwargs):

        super().__init__(adj, x, labels, device=device, seed=seed, name=name, **kwargs)

        self.norm_adj = norm_adj
        self.norm_x = norm_x
        self.preprocess(adj, x)
        self.do_before_train = self.extra_train

    def preprocess(self, adj, x):
        super().preprocess(adj, x)
        adj, x = self.adj, self.x

        if self.norm_adj:
            adj = normalize_adj(adj, self.norm_adj)

        if self.norm_x:
            x = normalize_x(x, norm=self.norm_x)

        with tf.device(self.device):
            self.x_norm, self.adj_norm = astensors([x, adj])

    def build(self, hiddens=[16], activations=['relu'], dropouts=[0.5], l2_norms=[5e-4],
              lr=0.01, p1=1.4, p2=0.7, use_bias=False):

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

            x = Input(batch_shape=[None, self.n_features], dtype=self.floatx, name='features')
            adj = Input(batch_shape=[None, None], dtype=self.floatx, sparse=True, name='adj_matrix')
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
            model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                          optimizer=Adam(lr=lr), metrics=['accuracy'])

            self.r_vadv = tf.Variable(TruncatedNormal(stddev=0.01)(shape=[self.n_nodes, self.n_features]), name="r_vadv")
            entropy_loss = entropy_y_x(logit)
            vat_loss = self.virtual_adversarial_loss(x, adj, logit)
            model.add_loss(p1 * vat_loss + p2 * entropy_loss)

            self.model = model
            self.adv_optimizer = Adam(lr=lr/10)

    def virtual_adversarial_loss(self, x, adj, logit):

        adv_x = x + self.r_vadv
        logit_p = tf.stop_gradient(logit)
        logit_m = self.propagation(adv_x, adj)
        loss = kl_divergence_with_logit(logit_p, logit_m)
        return loss

    def propagation(self, x, adj):
        h = x
        for dropout_layer, GCN_layer in zip(self.dropout_layers, self.GCN_layers[:-1]):
            h = dropout_layer(h)
            h = GCN_layer([h, adj])
        h = dropout_layer(h)
        h = self.GCN_layers[-1]([h, adj])
        return h

    @ tf.function
    def extra_train(self, epochs=10):

        with tf.device(self.device):
            r_vadv = self.r_vadv
            optimizer = self.adv_optimizer
            x, adj = self.x_norm, self.adj_norm
            r_vadv.assign(TruncatedNormal(stddev=0.01)(shape=tf.shape(r_vadv)))
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
        index = asintarr(index)

        with tf.device(self.device):
            logit = self.model.predict_on_batch([self.x_norm, self.adj_norm, index])

        if tf.is_tensor(logit):
            logit = logit.numpy()
        return logit

    def train_sequence(self, index):
        index = asintarr(index)
        labels = self.labels[index]

        with tf.device(self.device):
            sequence = FullBatchNodeSequence([self.x_norm, self.adj_norm, index], labels)

        return sequence
