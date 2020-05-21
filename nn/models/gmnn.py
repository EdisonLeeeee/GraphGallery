import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Softmax
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import regularizers

from graphgallery.nn.layers import GraphConvolution
from graphgallery.mapper import FullBatchNodeSequence
from graphgallery.nn.models.base import SupervisedModel


class GMNN(SupervisedModel):
    """
        Implementation of Graph Markov Neural Networks (GMNN). 
        [Graph Markov Neural Networks](https://arxiv.org/abs/1905.06214)
        Pytorch implementation: https://github.com/DeepGraphLearning/GMNN

        Arguments:
        ----------
            adj: `scipy.sparse.csr_matrix` (or `csc_matrix`) with shape (N, N)
                The input `symmetric` adjacency matrix, where `N` is the number of nodes 
                in graph.
            features: `np.array` with shape (N, F)
                The input node feature matrix, where `F` is the dimension of node features.
            labels: `np.array` with shape (N,)
                The ground-truth labels for all nodes in graph.
            normalize_rate (Float scalar, optional): 
                The normalize rate for adjacency matrix `adj`. (default: :obj:`-0.5`, 
                i.e., math:: \hat{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}}) 
            normalize_features (Boolean, optional): 
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

    def __init__(self, adj, features, labels, normalize_rate=-0.5, normalize_features=True, device='CPU:0', seed=None, **kwargs):

        super().__init__(adj, features, labels, device=device, seed=seed, **kwargs)

        self.normalize_rate = normalize_rate
        self.normalize_features = normalize_features
        self.preprocess(adj, features)
        self.labels_onehot = np.eye(self.n_classes)[labels]
        self.custom_objects = {'GraphConvolution': GraphConvolution}

    def preprocess(self, adj, features):
        super().preprocess(adj, features)

        if self.normalize_rate is not None:
            adj = self._normalize_adj(adj, self.normalize_rate)

        if self.normalize_features:
            features = self._normalize_features(features)

        with tf.device(self.device):
            self.features, self.adj = self._to_tensor([features, adj])

    def build(self, hidden_layers=[16], activations=['relu'], dropout=0.5,
              learning_rate=0.05, l2_norm=5e-4, use_bias=False):

        with tf.device(self.device):
            tf.random.set_seed(self.seed)
            x_p = Input(batch_shape=[self.n_nodes, self.n_classes], dtype=tf.float32, name='input_p')
            x_q = Input(batch_shape=[self.n_nodes, self.n_features], dtype=tf.float32, name='input_q')
            adj = Input(batch_shape=[self.n_nodes, self.n_nodes], dtype=tf.float32, sparse=True, name='adj_matrix')
            index = Input(batch_shape=[None],  dtype=tf.int64, name='index')

            def build_GCN(x):
                h = Dropout(rate=dropout)(x)

                for hid, activation in zip(hidden_layers, activations):
                    h = GraphConvolution(hid, use_bias=use_bias,
                                         activation=activation,
                                         kernel_regularizer=regularizers.l2(l2_norm))([h, adj])
#                     h = Dropout(rate=dropout)(h)

                h = GraphConvolution(self.n_classes, use_bias=use_bias)([h, adj])
                h = tf.ensure_shape(h, [self.n_nodes, self.n_classes])
                h = tf.gather(h, index)
                output = Softmax()(h)

                model = Model(inputs=[x, adj, index], outputs=output)
                model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=learning_rate), metrics=['accuracy'])
                return model

            # model_p
            model_p = build_GCN(x_p)

            # model_q
            model_q = build_GCN(x_q)

            self.model_p, self.model_q = model_p, model_q
            self.built = True

    def train(self, index_train, index_val=None, pre_train_epochs=100,
              epochs=100, early_stopping=None, validation=True,
              verbose=None, save_best=True, log_path=None, save_model=False,
              monitor_metric='val_accuracy', early_stop_metric='val_loss'):

        index_all = tf.range(self.n_nodes, dtype=tf.int64)

        # pre train model_q
        self.model = self.model_q
        super().train(index_train, index_val, epochs=pre_train_epochs,
                      early_stopping=early_stopping, validation=validation,
                      verbose=verbose, save_best=save_best, log_path=log_path, save_model=True,
                      monitor_metric=monitor_metric, early_stop_metric=early_stop_metric)

        label_predict = self.predict(index_all).argmax(1)
        label_predict[index_train] = self.labels[index_train]
        label_predict = tf.one_hot(label_predict, depth=self.n_classes)
        # train model_p fitst
        with tf.device(self.device):
            train_sequence = FullBatchNodeSequence([label_predict, self.adj, index_all], label_predict)
            if index_val is not None:
                val_sequence = FullBatchNodeSequence([label_predict, self.adj, index_val], self.labels_onehot[index_val])
            else:
                val_sequence = None
                
        self.model = self.model_p
        super().train(train_sequence, val_sequence, epochs=epochs,
                      early_stopping=early_stopping, validation=validation,
                      verbose=verbose, save_best=save_best, log_path=log_path, save_model=save_model,
                      monitor_metric=monitor_metric, early_stop_metric=early_stop_metric)

        # then train model_q again
        label_predict = self.model_p.predict_on_batch(self._to_tensor([label_predict, self.adj, index_all]))
        if tf.is_tensor(label_predict):
            label_predict = label_predict.numpy()
            
        label_predict[index_train] = self.labels_onehot[index_train]
        self.model = self.model_q
        with tf.device(self.device):
            train_sequence = FullBatchNodeSequence([self.features, self.adj, index_all], label_predict)
        history = super().train(train_sequence, index_val, epochs=epochs,
                                early_stopping=early_stopping, validation=validation,
                                verbose=verbose, save_best=save_best,
                                log_path=log_path, save_model=save_model,
                                monitor_metric=monitor_metric, early_stop_metric=early_stop_metric)

        return history

    def train_sequence(self, index):
        index = self._check_and_convert(index)
        labels = self.labels_onehot[index]
        with tf.device(self.device):
            sequence = FullBatchNodeSequence([self.features, self.adj, index], labels)
        return sequence

    def predict(self, index):
        super().predict(index)
        index = self._check_and_convert(index)
        with tf.device(self.device):
            index = self._to_tensor(index)
            logit = self.model.predict_on_batch([self.features, self.adj, index])

        if tf.is_tensor(logit):
            logit = logit.numpy()
        return logit
