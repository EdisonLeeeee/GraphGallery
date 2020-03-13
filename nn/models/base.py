import os
import random
import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence

from graphgallery.utils import History, sample_mask, normalize_adj, to_tensor, is_iterable


class SupervisedModel:

    def __init__(self, adj, features, labels, **kwargs):

        seed = kwargs.pop('seed', None)
        device = kwargs.pop('device', 'CPU:0')

        self.n_nodes, self.n_features = features.shape
        self.n_classes = labels.max() + 1
        
        self.adjacency_matrix = adj
        self.feature_matrix = features
        self.labels = labels

#         tf.config.set_soft_device_placement(True)

        if seed:
            tf.random.set_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        self.seed = seed
        self.device = tf.device(device)

        self.model = None
        self.built = None
        self.do_before_train = None
        self.do_before_validation = None
        self.do_before_test = None
        self.do_before_predict = None

        self.log_path = f'./log/{self.__class__.__name__}_weights.ckpt'

    def build(self):
        raise NotImplementedError

    def train(self, index_train, index_val=None,
              epochs=200, early_stopping=None, validation=True,
              verbose=None, restore_best=True, log_path=None,
              best_metric='val_accuracy', early_stop_metric='val_loss'):

        # Check if model has built
        if not self.built:
            raise RuntimeError('You must compile your model before training/testing/predicting. Use `model.build()`.')
            
            
        train_data = self.train_sequence(index_train)

        if validation and index_val is None:
            raise RuntimeError('`index_val` must be specified when `validation=True`.')
        
        if index_val is not None:
            val_data = self.test_sequence(index_val)

        history = History(best_metric=best_metric,
                          early_stop_metric=early_stop_metric)

        if log_path is None:
            log_path = self.log_path

        if not validation:
            history.register_best_metric('accuracy')
            history.register_early_stop_metric('loss')

        for epoch in range(1, epochs+1):

            if self.do_before_train is not None:
                self.do_before_train()

            loss, accuracy = self.do_forward(train_data)
            train_data.on_epoch_end()

            history.add_results(loss, 'loss')
            history.add_results(accuracy, 'accuracy')

            if validation:
                if self.do_before_validation is not None:
                    self.do_before_validation()

                val_loss, val_accuracy = self.do_forward(val_data, training=False)

                history.add_results(val_loss, 'val_loss')
                history.add_results(val_accuracy, 'val_accuracy')

            # record eoch and running times
            history.record_epoch(epoch)

            if restore_best and history.restore_best:
                self.save(log_path)

            # early stopping
            if early_stopping and history.time_to_early_stopping(early_stopping):
                if verbose:
                    print(f'Epoch {epoch}: early stopping with patience {early_stopping}.')
                break

            if verbose and history.times % verbose == 0:
                print(f'Epoch {epoch:3d}: loss {loss:.4f} - accuracy {accuracy:.2%}', end='')
                if validation:
                    print(f' - val_loss {val_loss:.4f} - val_accuracy {val_accuracy:.2%}.')
                else:
                    print()

        if restore_best:
            self.load(log_path)

        return history

    def test(self, index_test):

        if not self.built:
            raise RuntimeError('You must compile your model before training/testing/predicting. Use `model.build()`.')
            
        test_data = self.test_sequence(index_test)
        
        if self.do_before_test is not None:
            self.do_before_test()

        loss, accuracy = self.do_forward(test_data, training=False)

        return loss, accuracy

    def do_forward(self, data, training=True):

        if training:
            forward_fn = self.model.train_on_batch
        else:
            forward_fn = self.model.test_on_batch

        loss = accuracy = 0
        node_count_seen = 0

        with self.device:
            for inputs, labels in data:
                node_count = labels.shape[0]

                if node_count == 0:
                    continue

                loss_, accuracy_ = forward_fn(x=inputs, y=labels)

                loss += loss_ * node_count
                accuracy += accuracy_ * node_count
                node_count_seen += node_count

        if node_count_seen != 0:
            loss /= node_count_seen
            accuracy /= node_count_seen
        else:
            print('No nodes seen in forward!')

        return loss, accuracy

    def predict(self, index, **kwargs):
        
        if not self.built:
            raise RuntimeError('You must compile your model before training/testing/predicting. Use `model.build()`.')
            
        if self.do_before_predict is not None:
            self.do_before_predict(index, **kwargs)
            
    def train_sequence(self, index):
        raise NotImplementedError
        
    def test_sequence(self, index):
        return self.train_sequence(index) 
    
    def test_predict(self, index, labels):
        logit = self.predict(index)
        predict_class = logit.argmax(1)
        return (predict_class == labels).mean()

    @staticmethod
    def _to_tensor(inputs):
        """Convert input matrices to Tensor (SparseTensor)."""
        return to_tensor(inputs)

    @staticmethod
    def _normalize_adj(adj, rate=-0.5, self_loop=True):
        """Normalize adjacency matrix."""
        return normalize_adj(adj, rate=rate, self_loop=self_loop)
    
    @staticmethod    
    def _normalize_features(features):
        assert isinstance(features, np.ndarray)
        return features / (features.sum(1, keepdims=True) + 1e-10)

    def _sample_mask(self, index, shape=None):
        if shape is None:
            shape = self.n_nodes
        return sample_mask(index, shape)
    
    @staticmethod
    def _is_iterable(arr):
        return is_iterable(arr)

    @staticmethod
    def _check_and_convert(index):
        if isinstance(index, int):
            index = np.asarray([index])
        elif isinstance(index, list):
            index = np.asarray(index)
        elif isinstance(index, np.ndarray):
            pass
        else:
            raise ValueError('`index` should be either list, int or np.ndarray!')
        return index

    def save(self, path=None):
        if not os.path.exists('log'):
            os.makedirs('log')       
            
        if not path:
            path = self.log_path
        self.model.save_weights(path)

    def load(self, path=None):
        if not path:
            path = self.log_path
        self.model.load_weights(path)

    @property
    def weights(self):
        return self.model.weights

    @property
    def np_weights(self):
        weights = []
        for W in self.weights:
            weights.append(W.numpy())
        return weights

    @property
    def close(self):
        #         del self.model
        self.built = None
        K.clear_session()

        
        
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

class UnsupervisedModel:

    def __init__(self, adj, features, labels, **kwargs):

        seed = kwargs.pop('seed', None)
        device = kwargs.pop('device', 'CPU:0')

        self.n_nodes, self.n_features = features.shape
        self.n_classes = labels.max() + 1
        self.adj, self.features = adj, features
        self.labels = labels

        if seed:
            tf.random.set_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        self.seed = seed
        self.device = tf.device(device)

        self.log_path = f'./log/{self.__class__.__name__}_weights.ckpt'     
        self.embeddings = None
        self.model = None
        
        self.clssifier = LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='auto', random_state=seed)

    def build(self):
        raise NotImplementedError

                
    def get_embeddings(self):
        raise NotImplementedError

    def train(self, index):
        if self.embeddings is None:
            self.get_embeddings()
            
        index = self._check_and_convert(index)
        self.clssifier.fit(self.embeddings[index], self.labels[index])

    def predict(self, index):
        index = self._check_and_convert(index)
        logit = self.clssifier.predict_proba(self.embeddings[index])
        return logit
    
    def test(self, index):
        index = self._check_and_convert(index)
        y_true =  self.labels[index]
        y_pred = self.clssifier.predict(self.embeddings[index])
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy
    
    @staticmethod
    def _normalize_embedding(embeddings):
        return normalize(embeddings)
    
    @staticmethod
    def _check_and_convert(index):
        if isinstance(index, int):
            index = np.asarray([index])
        elif isinstance(index, list):
            index = np.asarray(index)
        elif isinstance(index, np.ndarray):
            pass
        else:
            raise ValueError('`index` should be either list, int or np.ndarray!')
        return index        
