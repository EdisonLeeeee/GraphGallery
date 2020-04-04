import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from graphgallery.nn.layers import GaussionConvolution_F, GaussionConvolution_D
from graphgallery.mapper import FullBatchNodeSequence
from .base import SupervisedModel


class RobustGCN(SupervisedModel):
    """
        Implementation of Robust Graph Convolutional Networks (RobustGCN). 
        [Robust Graph Convolutional Networks Against Adversarial Attacks](https://dl.acm.org/doi/10.1145/3292500.3330851)
        Tensorflow 1.x implementation: https://github.com/thumanlab/nrlweb/blob/master/static/assets/download/RGCN.zip

        Arguments:
        ----------
            adj: `scipy.sparse.csr_matrix` (or `csr_matrix`) with shape (N, N), the input `symmetric` adjacency matrix, where `N` is the number of nodes in graph.
            features: `np.array` with shape (N, F), the input node feature matrix, where `F` is the dimension of node features.
            labels: `np.array` with shape (N,), the ground-truth labels for all nodes in graph.
            normalize_rate (List of float scalar, optional): The normalize rate for adjacency matrix `adj`. (default: :obj:`[-0.5, -1]`, i.e., two normalized `adj` with rate `-0.5` and `-1.0`, respectively) 
            normalize_features (Boolean, optional): Whether to use row-normalize for node feature matrix. (default :obj: `True`)
            device (String, optional): The device where the model is running on. You can specified `CPU` or `GPU` for the model. (default: :obj: `CPU:0`, i.e., the model is running on the 0-th device `CPU`)
            seed (Positive integer, optional): Used in combination with `tf.random.set_seed & np.random.seed & random.seed` to create a reproducible sequence of tensors across multiple calls. (default :obj: `None`, i.e., using random seed)

    """    
    
    def __init__(self, adj, features, labels, normalize_rate=[-0.5, -1], normalize_features=True, device='CPU:0', seed=None):
    
        super().__init__(adj, features, labels, device=device, seed=seed)
        
        self.normalize_rate = normalize_rate
        self.normalize_features = normalize_features            
        self.preprocess(adj, features)
            
    def preprocess(self, adj, features):
        
        if self.normalize_rate is not None:
            adj = self._normalize_adj([adj, adj], self.normalize_rate)    # [adj_1, adj_2]    
            
        if self.normalize_features:
            features = self._normalize_features(features)
            
        with self.device:
            self.features, self.adj = self._to_tensor([features, adj])
        
    def build(self, hidden_layers=[64], activations=['relu'], use_bias=False, dropout=0.6, learning_rate=0.01, l2_norm=5e-4, para_kl=5e-4, gamma=1.0):
        
        x = Input(batch_shape=[self.n_nodes, self.n_features], dtype=tf.float32, name='features')
        adj = [Input(batch_shape=[self.n_nodes, self.n_nodes], dtype=tf.float32, sparse=True, name='adj_matrix_1'),
               Input(batch_shape=[self.n_nodes, self.n_nodes], dtype=tf.float32, sparse=True, name='adj_matrix_2')]
        index = Input(batch_shape=[None],  dtype=tf.int32, name='index')

        h = Dropout(rate=dropout)(x)
        h, KL_divergence = GaussionConvolution_F(hidden_layers[0], gamma=gamma, 
                                                 use_bias=use_bias, 
                                                 activation=activations[0], 
                                                 kernel_regularizer=regularizers.l2(l2_norm))([h, *adj])
        
        # additional layers (usually unnecessay)
        for hid, activation in zip(hidden_layers[1:], activations[1:]):
            h = Dropout(rate=dropout)(h)
            h = GaussionConvolution_D(hid, gamma=gamma, use_bias=use_bias, activation=activation)([h, *adj])
            
        h = Dropout(rate=dropout)(h)
        h = GaussionConvolution_D(self.n_classes, gamma=gamma, use_bias=use_bias)([h, *adj])
        h = tf.ensure_shape(h, [self.n_nodes, self.n_classes])            
        h = tf.gather(h, index)
        output = Softmax()(h)

        model = Model(inputs=[x, *adj, index], outputs=output)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
        model.add_loss(para_kl * KL_divergence)
        
        self.model = model
        self.built = True
        
    def train_sequence(self, index):
        index = self._check_and_convert(index)
        labels = self.labels[index]      
        with self.device:
            sequence = FullBatchNodeSequence([self.features, *self.adj, index], labels)
        return sequence

    def predict(self, index):
        if not self.built:
            raise RuntimeError('You must compile your model before training/testing/predicting. Use `model.build()`.')

        if self.do_before_predict is not None:
            self.do_before_predict(idx, **kwargs)

        index = self._check_and_convert(index)

        with self.device:
            index = self._to_tensor(index)
            logit = self.model.predict_on_batch([self.features, *self.adj, index])

        return logit.numpy()    