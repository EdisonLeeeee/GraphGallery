import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import TruncatedNormal

from graphgallery.nn.layers import GraphConvolution
from graphgallery.nn.models import SemiSupervisedModel
from graphgallery.sequence import FullBatchNodeSequence
from graphgallery.utils.bvat_utils import kl_divergence_with_logit, entropy_y_x, get_normalized_vector
from graphgallery.utils.shape_utils import set_equal_in_length
from graphgallery import astensor, asintarr, normalize_x, normalize_adj, Bunch


class SimplifiedOBVAT(SemiSupervisedModel):
    """
        Implementation of optimization-based Batch Virtual Adversarial Training  Graph Convolutional Networks (OBVAT). 
        `Batch Virtual Adversarial Training for Graph Convolutional Networks <https://arxiv.org/abs/1902.09192>`
        Tensorflow 1.x implementation: <https://github.com/thudzj/BVAT>

        Note:
        ----------
        This is a simplified implementation of `OBVAT`.


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
            norm_adj_rate (Float scalar, optional): 
                The normalize rate for adjacency matrix `adj`. (default: :obj:`-0.5`, 
                i.e., math:: \hat{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}}) 
            norm_x_type (String, optional): 
                How to normalize the node feature matrix. See `graphgallery.normalize_x`
                (default :str: `l1`)
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

    def __init__(self, adj, x, labels, norm_adj_rate=-0.5, norm_x_type='l1',
                 device='CPU:0', seed=None, name=None, **kwargs):

        super().__init__(adj, x, labels, device=device, seed=seed, name=name, **kwargs)

        self.norm_adj_rate = norm_adj_rate
        self.norm_x_type = norm_x_type
        self.preprocess(adj, x)

    def preprocess(self, adj, x):
        super().preprocess(adj, x)
        # check the input adj and x, and convert them into proper data types
        adj, x = self._check_inputs(adj, x)

        if self.norm_adj_rate:
            adj = normalize_adj(adj, self.norm_adj_rate)

        if self.norm_x_type:
            x = normalize_x(x, norm=self.norm_x_type)

        with tf.device(self.device):
            self.x_norm, self.adj_norm = astensor([x, adj])

    def build(self, hiddens=[16], activations=['relu'], dropouts=[0.5],
              lr=0.01, l2_norms=[5e-4], p1=1.4, p2=0.7,
              epsilon=0.01, ensure_shape=True):

        local_paras = locals()
        local_paras.pop('self')
        paras = Bunch(**local_paras)
        hiddens, activations, dropouts, l2_norms = set_equal_in_length(hiddens, activations, dropouts, l2_norms)
        paras.update(Bunch(hiddens=hiddens, activations=activations, dropouts=dropouts, l2_norms=l2_norms))
        # update all parameters
        self.paras.update(paras)
        self.model_paras.update(paras)

        with tf.device(self.device):

            x = Input(batch_shape=[None, self.n_features], dtype=self.floatx, name='features')
            adj = Input(batch_shape=[None, None], dtype=self.floatx, sparse=True, name='adj_matrix')
            index = Input(batch_shape=[None],  dtype=self.intx, name='index')

            GCN_layers = []
            dropout_layers = []
            for hid, activation, dropout, l2_norm in zip(hiddens, activations, dropouts, l2_norms):
                GCN_layers.append(GraphConvolution(hid, activation=activation,
                                                   kernel_regularizer=regularizers.l2(l2_norm)))
                dropout_layers.append(Dropout(rate=dropout))

            GCN_layers.append(GraphConvolution(self.n_classes))
            self.GCN_layers = GCN_layers
            self.dropout_layers = dropout_layers

            logit = self.propagation(x, adj)
            # To aviod the UserWarning of `tf.gather`, but it causes the shape
            # of the input data to remain the same
            if ensure_shape:
                logit = tf.ensure_shape(logit, (self.n_nodes, self.n_classes))
            output = tf.gather(logit, index)
            output = Softmax()(output)
            model = Model(inputs=[x, adj, index], outputs=output)
            model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])

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
            index = astensor(index)
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
