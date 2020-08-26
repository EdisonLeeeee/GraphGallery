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
from graphgallery.utils.bvat_utils import kl_divergence_with_logit, entropy_y_x, get_normalized_vector
from graphgallery.utils.shape import set_equal_in_length
from graphgallery import astensors, asintarr, normalize_x, normalize_adj, Bunch


class SimplifiedOBVAT(SemiSupervisedModel):
    """
        Implementation of optimization-based Batch Virtual Adversarial Training  Graph Convolutional Networks (OBVAT). 
        `Batch Virtual Adversarial Training for Graph Convolutional Networks <https://arxiv.org/abs/1902.09192>`
        Tensorflow 1.x implementation: <https://github.com/thudzj/BVAT>


    """

    def __init__(self, adj, x, labels, norm_adj=-0.5, norm_x=None,
                 device='CPU:0', seed=None, name=None, **kwargs):
        """Creat a Simplified OBVAT model.

        Parameters:
        ----------
            adj: Scipy.sparse.csr_matrix, shape [n_nodes, n_nodes]
                The input `symmetric` adjacency matrix in CSR format.
            x: Numpy.ndarray, shape [n_nodes, n_attrs]. 
                Node attribute matrix in Numpy format.
            labels: Numpy.ndarray, shape [n_nodes]
                Array, where each entry represents respective node's label(s).
            norm_adj: float scalar. optional 
                The normalize rate for adjacency matrix `adj`. (default: :obj:`-0.5`, 
                i.e., math:: \hat{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}}) 
            norm_x: string. optional 
                How to normalize the node attribute matrix. See `graphgallery.normalize_x`
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

        Note:
        ----------
        This is a simplified implementation of `OBVAT`.                
        """
        super().__init__(adj, x, labels, device=device, seed=seed, name=name, **kwargs)

        self.norm_adj = norm_adj
        self.norm_x = norm_x
        self.preprocess(adj, x)

    def preprocess(self, adj, x):
        super().preprocess(adj, x)
        adj, x = self.adj, self.x

        if self.norm_adj:
            adj = normalize_adj(adj, self.norm_adj)

        if self.norm_x:
            x = normalize_x(x, norm=self.norm_x)

        with tf.device(self.device):
            self.x_norm, self.adj_norm = astensors([x, adj])

    def build(self, hiddens=[16], activations=['relu'], dropouts=[0.],
              lr=0.01, l2_norms=[5e-4], p1=1.4, p2=0.7, use_bias=False,
              epsilon=0.01):

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

            x = Input(batch_shape=[None, self.n_attrs], dtype=self.floatx, name='attributes')
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

            entropy_loss = entropy_y_x(logit)
            vat_loss = self.virtual_adversarial_loss(x, adj, logit, epsilon)
            model.add_loss(p1 * vat_loss + p2 * entropy_loss)

            self.model = model
            self.adv_optimizer = Adam(lr=lr/10)

    def virtual_adversarial_loss(self, x, adj, logit, epsilon):
        d = tf.random.normal(shape=tf.shape(x), dtype=self.floatx)

        r_vadv = get_normalized_vector(d) * epsilon
        logit_p = tf.stop_gradient(logit)
        logit_m = self.propagation(x + r_vadv, adj)
        loss = kl_divergence_with_logit(logit_p, logit_m)
        return tf.identity(loss)

    def propagation(self, x, adj):
        h = x
        for dropout_layer, GCN_layer in zip(self.dropout_layers, self.GCN_layers[:-1]):
            h = GCN_layer([h, adj])
            h = dropout_layer(h)
        h = self.GCN_layers[-1]([h, adj])
        return h

    def predict(self, index):
        super().predict(index)
        index = asintarr(index)

        with tf.device(self.device):
            index = astensors(index)
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
