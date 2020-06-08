import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from graphgallery.nn.layers import GaussionConvolution_F, GaussionConvolution_D
from graphgallery.sequence import FullBatchNodeSequence
from graphgallery.nn.models import SupervisedModel


class RobustGCN(SupervisedModel):
    """
        Implementation of Robust Graph Convolutional Networks (RobustGCN). 
        [Robust Graph Convolutional Networks Against Adversarial Attacks](https://dl.acm.org/doi/10.1145/3292500.3330851)
        Tensorflow 1.x implementation: https://github.com/thumanlab/nrlweb/blob/master/static/assets/download/RGCN.zip

        Arguments:
        ----------
            adj: `scipy.sparse.csr_matrix` (or `csc_matrix`) with shape (N, N)
                The input `symmetric` adjacency matrix, where `N` is the number of nodes 
                in graph.
            x: `np.array` with shape (N, F)
                The input node feature matrix, where `F` is the dimension of node features.
            labels: `np.array` with shape (N,)
                The ground-truth labels for all nodes in graph.
            normalize_rate (List of float scalar, optional): 
                The normalize rate for adjacency matrix `adj`. 
                (default: :obj:`[-0.5, -1]`, i.e., two normalized `adj` with rate `-0.5` 
                and `-1.0`, respectively) 
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

    def __init__(self, adj, x, labels, normalize_rate=[-0.5, -1], is_normalize_x=True, 
                 device='CPU:0', seed=None, name=None, **kwargs):

        super().__init__(adj, x, labels, device=device, seed=seed, name=name, **kwargs)

        self.normalize_rate = normalize_rate
        self.is_normalize_x = is_normalize_x
        self.preprocess(adj, x)

    def preprocess(self, adj, x):
        adj, x = super().preprocess(adj, x)

        if self.normalize_rate is not None:
            adj = self.normalize_adj([adj, adj], self.normalize_rate)    # [adj_1, adj_2]

        if self.is_normalize_x:
            x = self.normalize_x(x)

        with tf.device(self.device):
            self.tf_x, self.tf_adj = self.to_tensor([x, adj])

    def build(self, hiddens=[64], activations=['relu'], use_bias=False, dropout=0.6, lr=0.01, l2_norm=1e-4, para_kl=5e-4, gamma=1.0, ensure_shape=True):
        
        assert len(hiddens) == len(activations)
        
        with tf.device(self.device):

            x = Input(batch_shape=[None, self.n_features], dtype=self.floatx, name='features')
            adj = [Input(batch_shape=[None, None], dtype=self.floatx, sparse=True, name='adj_matrix_1'),
                   Input(batch_shape=[None, None], dtype=self.floatx, sparse=True, name='adj_matrix_2')]
            index = Input(batch_shape=[None],  dtype=self.intx, name='index')

            h = Dropout(rate=dropout)(x)
            h, KL_divergence = GaussionConvolution_F(hiddens[0], gamma=gamma,
                                                     use_bias=use_bias,
                                                     activation=activations[0],
                                                     kernel_regularizer=regularizers.l2(l2_norm))([h, *adj])

            # additional layers (usually unnecessay)
            for hid, activation in zip(hiddens[1:], activations[1:]):
                h = Dropout(rate=dropout)(h)
                h = GaussionConvolution_D(hid, gamma=gamma, use_bias=use_bias, activation=activation)([h, *adj])

            h = Dropout(rate=dropout)(h)
            h = GaussionConvolution_D(self.n_classes, gamma=gamma, use_bias=use_bias)([h, *adj])
            if ensure_shape:
                h = tf.ensure_shape(h, [self.n_nodes, self.n_classes])
            h = tf.gather(h, index)
            output = Softmax()(h)

            model = Model(inputs=[x, *adj, index], outputs=output)
            model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])
            model.add_loss(para_kl * KL_divergence)

            self.set_model(model)
            self.built = True

    def train_sequence(self, index):
        index = self.to_int(index)
        labels = self.labels[index]
        with tf.device(self.device):
            sequence = FullBatchNodeSequence([self.tf_x, *self.tf_adj, index], labels)
        return sequence

    def predict(self, index):
        super().predict(index)
        index = self.to_int(index)

        with tf.device(self.device):
            index = self.to_tensor(index)
            logit = self.model.predict_on_batch([self.tf_x, *self.tf_adj, index])

        if tf.is_tensor(logit):
            logit = logit.numpy()
        return logit
