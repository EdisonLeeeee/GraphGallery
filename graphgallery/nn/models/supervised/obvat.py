import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import TruncatedNormal

from graphgallery.nn.layers import GraphConvolution
from graphgallery.sequence import FullBatchNodeSequence
from graphgallery.nn.models import SupervisedModel
from graphgallery.utils.bvat_utils import kl_divergence_with_logit, entropy_y_x


class OBVAT(SupervisedModel):
    """
        Implementation of optimization-based Batch Virtual Adversarial Training  Graph Convolutional Networks (OBVAT). 
        [Batch Virtual Adversarial Training for Graph Convolutional Networks](https://arxiv.org/abs/1902.09192)
        Tensorflow 1.x implementation: https://github.com/thudzj/BVAT

        Arguments:
        ----------
            adj: `scipy.sparse.csr_matrix` (or `csc_matrix`) with shape (N, N)
                The input `symmetric` adjacency matrix, where `N` is the number of nodes 
                in graph.
            x: `np.array` with shape (N, F)
                The input node feature matrix, where `F` is the dimension of node features.
            labels: `np.array` with shape (N,)
                The ground-truth labels for all nodes in graph.
            normalize_rate (Float scalar, optional): 
                The normalize rate for adjacency matrix `adj`. (default: :obj:`-0.5`, 
                i.e., math:: \hat{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}}) 
            is_normalize_x (Boolean, optional): 
                Whether to use row-normalize for node feature matrix. 
                (default :obj: `True`)
            device (String, optional): 
                The device where the model is running on. You can specified `CPU` or `GPU` 
                for the model. (default: :obj: `CPU:0`, i.e., the model is running on 
                the 0-th device `CPU`)
            seed (Positive integer, optional): 
                Used in combination with `tf.random.set_seed & np.random.seed & random.seed` 
                to create a reproducible sequence of tensors across multiple calls. 
                (default :obj: `None`, i.e., using random seed)
            name (String, optional): 
                Name for the model. (default: name of class)


    """

    def __init__(self, adj, x, labels, normalize_rate=-0.5, is_normalize_x=True, 
                 device='CPU:0', seed=None, name=None, **kwargs):

        super().__init__(adj, x, labels, device=device, seed=seed, name=name, **kwargs)

        self.normalize_rate = normalize_rate
        self.is_normalize_x = is_normalize_x
        self.preprocess(adj, x)
        self.do_before_train = self.extra_train

    def preprocess(self, adj, x):
        adj, x = super().preprocess(adj, x)

        if self.normalize_rate is not None:
            adj = self.normalize_adj(adj, self.normalize_rate)

        if self.is_normalize_x:
            x = self.normalize_x(x)

        with tf.device(self.device):
            self.tf_x, self.tf_adj = self.to_tensor([x, adj])

    def build(self, hiddens=[16], activations=['relu'], dropout=0.5, lr=0.01, l2_norm=5e-4, p1=1.4, p2=0.7, ensure_shape=True):

        with tf.device(self.device):

            x = Input(batch_shape=[None, self.n_features], dtype=self.floatx, name='features')
            adj = Input(batch_shape=[None, None], dtype=self.floatx, sparse=True, name='adj_matrix')
            index = Input(batch_shape=[None],  dtype=self.intx, name='index')

            self.GCN_layers = [GraphConvolution(hiddens[0], activation=activations[0],
                                                kernel_regularizer=regularizers.l2(l2_norm)),
                               GraphConvolution(self.n_classes)]
            self.dropout_layer = Dropout(rate=dropout)

            logit = self.propagation(x, adj)
            if ensure_shape:
                logit = tf.ensure_shape(logit, (self.n_nodes, self.n_classes))
            output = tf.gather(logit, index)
            output = Softmax()(output)
            model = Model(inputs=[x, adj, index], outputs=output)
            model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])

            self.r_vadv = tf.Variable(TruncatedNormal(stddev=0.01)(shape=[self.n_nodes, self.n_features]), name="r_vadv")
            entropy_loss = entropy_y_x(logit)
            vat_loss = self.virtual_adversarial_loss(x, adj, logit)
            model.add_loss(p1 * vat_loss + p2 * entropy_loss)

            self.set_model(model)
            self.adv_optimizer = Adam(lr=lr/10)
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

        with tf.device(self.device):
            r_vadv = self.r_vadv
            optimizer = self.adv_optimizer
            x, adj = self.tf_x, self.tf_adj
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
        index = self.to_int(index)

        with tf.device(self.device):
            index = self.to_tensor(index)
            logit = self.model.predict_on_batch([self.tf_x, self.tf_adj, index])

        if tf.is_tensor(logit):
            logit = logit.numpy()
        return logit

    def train_sequence(self, index):
        index = self.to_int(index)
        labels = self.labels[index]

        with tf.device(self.device):
            sequence = FullBatchNodeSequence([self.tf_x, self.tf_adj, index], labels)

        return sequence
