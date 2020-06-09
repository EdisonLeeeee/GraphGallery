import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Softmax
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import regularizers

from graphgallery.nn.layers import GraphConvolution
from graphgallery.sequence import FullBatchNodeSequence
from graphgallery.nn.models import SupervisedModel
from graphgallery.utils.data_utils import normalize_fn, normalize_adj


class GMNN(SupervisedModel):
    """
        Implementation of Graph Markov Neural Networks (GMNN). 
        [Graph Markov Neural Networks](https://arxiv.org/abs/1905.06214)
        Pytorch implementation: https://github.com/DeepGraphLearning/GMNN

        Arguments:
        ----------
            adj: shape (N, N), `scipy.sparse.csr_matrix` (or `csc_matrix`) if 
                `is_adj_sparse=True`, `np.array` or `np.matrix` if `is_adj_sparse=False`.
                The input `symmetric` adjacency matrix, where `N` is the number 
                of nodes in graph.
            x: shape (N, F), `scipy.sparse.csr_matrix` (or `csc_matrix`) if 
                `is_x_sparse=True`, `np.array` or `np.matrix` if `is_x_sparse=False`.
                The input node feature matrix, where `F` is the dimension of features.
            labels: `np.array` with shape (N,)
                The ground-truth labels for all nodes in graph.
            norm_adj_rate (Float scalar, optional): 
                The normalize rate for adjacency matrix `adj`. (default: :obj:`-0.5`, 
                i.e., math:: \hat{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}}) 
            norm_x_type (String, optional): 
                How to normalize the node feature matrix. See graphgallery.utils.normalize_fn
                (default :obj: `row_wise`)
            device (String, optional): 
                The device where the model is running on. You can specified `CPU` or `GPU` 
                for the model. (default: :obj: `CPU:0`, i.e., the model is running on 
                the 0-th device `CPU`)
            seed (Positive integer, optional): 
                Used in combination with `tf.random.set_seed` & `np.random.seed` & `random.seed`  
                to create a reproducible sequence of tensors across multiple calls. 
                (default :obj: `None`, i.e., using random seed)
            name (String, optional): 
                Specified name for the model. (default: `class.__name__`)

    """

    def __init__(self, adj, x, labels, norm_adj_rate=-0.5, norm_x_type='row_wise', 
                 device='CPU:0', seed=None, name=None, **kwargs):

        super().__init__(adj, x, labels, device=device, seed=seed, name=name, **kwargs)

        self.norm_adj_rate = norm_adj_rate
        self.norm_x_fn = normalize_fn(norm_x_type)
        self.preprocess(adj, x)
        self.labels_onehot = np.eye(self.n_classes)[labels]
        self.custom_objects = {'GraphConvolution': GraphConvolution}

    def preprocess(self, adj, x):
        adj, x = super().preprocess(adj, x)

        if self.norm_adj_rate is not None:
            adj = normalize_adj(adj, self.norm_adj_rate)

        if self.norm_x_fn is not None:
            x = self.norm_x_fn(x)

        with tf.device(self.device):
            self.tf_x, self.tf_adj = self.to_tensor([x, adj])

    def build(self, hiddens=[16], activations=['relu'], dropout=0.5,
              lr=0.05, l2_norm=5e-4, use_bias=False, ensure_shape=True):

        assert len(hiddens) == len(activations), "The number of hidden units and " \
                                                "activation function should be the same"
        
        with tf.device(self.device):
            tf.random.set_seed(self.seed)
            x_p = Input(batch_shape=[None, self.n_classes], dtype=self.floatx, name='input_p')
            x_q = Input(batch_shape=[None, self.n_features], dtype=self.floatx, name='input_q')
            adj = Input(batch_shape=[None, None], dtype=self.floatx, sparse=True, name='adj_matrix')
            index = Input(batch_shape=[None],  dtype=self.intx, name='index')

            def build_GCN(x):
                h = Dropout(rate=dropout)(x)

                for hid, activation in zip(hiddens, activations):
                    h = GraphConvolution(hid, use_bias=use_bias,
                                         activation=activation,
                                         kernel_regularizer=regularizers.l2(l2_norm))([h, adj])
#                     h = Dropout(rate=dropout)(h)

                h = GraphConvolution(self.n_classes, use_bias=use_bias)([h, adj])
                # To aviod the UserWarning of `tf.gather`, but it causes the shape 
                # of the input data to remain the same
                if ensure_shape:
                        h = tf.ensure_shape(h, [self.n_nodes, self.n_classes])
                h = tf.gather(h, index)
                output = Softmax()(h)

                model = Model(inputs=[x, adj, index], outputs=output)
                model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=lr), metrics=['accuracy'])
                return model

            # model_p
            model_p = build_GCN(x_p)

            # model_q
            model_q = build_GCN(x_q)

            self.model_p, self.model_q = model_p, model_q
            self.set_model(self.model_q)
            self.built = True

    def train(self, index_train, index_val=None, pre_train_epochs=100,
              epochs=100, early_stopping=None,
              verbose=None, save_best=True, log_path=None, save_model=False,
              monitor='val_acc', early_stop_metric='val_loss'):

        index_all = tf.range(self.n_nodes, dtype=self.intx)

        # pre train model_q
        self.set_model(self.model_q)
        super().train(index_train, index_val, epochs=pre_train_epochs,
                      early_stopping=early_stopping,
                      verbose=verbose, save_best=save_best, log_path=log_path, save_model=True,
                      monitor=monitor, early_stop_metric=early_stop_metric)

        label_predict = self.predict(index_all).argmax(1)
        label_predict[index_train] = self.labels[index_train]
        label_predict = tf.one_hot(label_predict, depth=self.n_classes)
        # train model_p fitst
        with tf.device(self.device):
            train_sequence = FullBatchNodeSequence([label_predict, self.tf_adj, index_all], label_predict)
            if index_val is not None:
                val_sequence = FullBatchNodeSequence([label_predict, self.tf_adj, index_val], self.labels_onehot[index_val])
            else:
                val_sequence = None

        self.set_model(self.model_p)
        super().train(train_sequence, val_sequence, epochs=epochs,
                      early_stopping=early_stopping,
                      verbose=verbose, save_best=save_best, log_path=log_path, save_model=save_model,
                      monitor=monitor, early_stop_metric=early_stop_metric)

        # then train model_q again
        label_predict = self.model.predict_on_batch(self.to_tensor([label_predict, self.tf_adj, index_all]))
        if tf.is_tensor(label_predict):
            label_predict = label_predict.numpy()

        label_predict[index_train] = self.labels_onehot[index_train]
        
        self.set_model(self.model_q)
        with tf.device(self.device):
            train_sequence = FullBatchNodeSequence([self.tf_x, self.tf_adj, index_all], label_predict)
        history = super().train(train_sequence, index_val, epochs=epochs,
                                early_stopping=early_stopping,
                                verbose=verbose, save_best=save_best,
                                log_path=log_path, save_model=save_model,
                                monitor=monitor, early_stop_metric=early_stop_metric)

        return history

    def train_sequence(self, index):
        index = self.to_int(index)
        labels = self.labels_onehot[index]
        with tf.device(self.device):
            sequence = FullBatchNodeSequence([self.tf_x, self.tf_adj, index], labels)
        return sequence

    def predict(self, index):
        super().predict(index)
        index = self.to_int(index)
        with tf.device(self.device):
            index = self.to_tensor(index)
            logit = self.model.predict_on_batch([self.tf_x, self.tf_adj, index])

        if tf.is_tensor(logit):
            logit = logit.numpy()
        return logit
