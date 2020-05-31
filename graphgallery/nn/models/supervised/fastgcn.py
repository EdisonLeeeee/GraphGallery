import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from graphgallery.nn.layers import GraphConvolution
from graphgallery.sequence import FastGCNBatchSequence
from graphgallery.nn.models import SupervisedModel


class FastGCN(SupervisedModel):
    """
        Implementation of Fast Graph Convolutional Networks (FastGCN). 
        [FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling](https://arxiv.org/abs/1801.10247)
        Tensorflow 1.x implementation: https://github.com/matenure/FastGCN

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
            batch_size (Positive integer, optional): 
                Batch size for the training nodes. (default :obj: `256`)
            rank (Positive integer, optional): 
                The selected nodes for each batch nodes, `rank` must be smaller than 
                `batch_size`. (default :obj: `100`)
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

    def __init__(self, adj, x, labels, normalize_rate=-0.5, is_normalize_x=False,
                 batch_size=256, rank=100, device='CPU:0', seed=None, name=None, **kwargs):

        super().__init__(adj, x, labels, device=device, seed=seed, name=name, **kwargs)

        self.rank = rank
        self.batch_size = batch_size
        self.normalize_rate = normalize_rate
        self.is_normalize_x = is_normalize_x
        self.preprocess(adj, x)

    def preprocess(self, adj, x):
        adj, x = super().preprocess(adj, x)

        if self.normalize_rate is not None:
            adj = self.normalize_adj(adj, self.normalize_rate)

        if self.is_normalize_x:
            x = self.normalize_x(x)

        x = adj.dot(x)

        with tf.device(self.device):
            self.tf_x, self.adj_norm = self.to_tensor(x), adj

    def build(self, hiddens=[32], activations=['relu'], dropout=0.5,
              lr=0.01, l2_norm=5e-4, use_bias=False):
        
        assert len(hiddens) == len(activations)
        with tf.device(self.device):

            x = Input(batch_shape=[None, self.n_features], dtype=self.floatx, name='features')
            adj = Input(batch_shape=[None, None], dtype=self.floatx, sparse=True, name='adj_matrix')

            h = x
            for hid, activation in zip(hiddens, activations):
                h = Dense(hid, use_bias=use_bias, activation=activation, kernel_regularizer=regularizers.l2(l2_norm))(h)
                h = Dropout(rate=dropout)(h)

            output = GraphConvolution(self.n_classes, activation='softmax')([h, adj])

            model = Model(inputs=[x, adj], outputs=output)
            model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])

            self.set_model(model)
            self.built = True

    def predict(self, index):
        super().predict(index)
        index = self.to_int(index)
        adj = self.adj_norm[index]
        with tf.device(self.device):
            adj = self.to_tensor(adj)
            logit = self.model.predict_on_batch([self.tf_x, adj])

        if tf.is_tensor(logit):
            logit = logit.numpy()
        return logit

    def train_sequence(self, index):
        index = self.to_int(index)
        labels = self.labels[index]
        adj = self.adj[index].tocsc()[:, index]

        if self.normalize_rate is not None:
            adj = self.normalize_adj(adj, self.normalize_rate)

        with tf.device(self.device):
            x = tf.gather(self.tf_x, index)
            sequence = FastGCNBatchSequence([x, adj], labels,
                                            batch_size=self.batch_size,
                                            rank=self.rank)
        return sequence

    def test_sequence(self, index):
        index = self.to_int(index)
        labels = self.labels[index]
        adj = self.adj_norm[index]

        with tf.device(self.device):
            sequence = FastGCNBatchSequence([self.tf_x, adj],
                                            labels, batch_size=None, rank=None)  # use full batch
        return sequence
