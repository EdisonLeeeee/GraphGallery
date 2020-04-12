import os
import random
import datetime
import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from tqdm import tqdm
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence

from graphgallery.utils import History, sample_mask, normalize_adj, to_tensor, is_iterable

# def printbar(epoch, total,bar_len=44):
#     t = datetime.datetime.now()
#     left = int((epoch/total)*bar_len)
#     right = bar_len - left
#     bar = '[' + "="*left + f'>' + '.'*right + ']'
#     bar = bar[:bar_len//2] + f' {epoch}/{total} ' + bar[bar_len//2:]
#     bar += f' {t.hour:02}:{t.minute:02}:{t.second:02}'
#     return bar
    
class SupervisedModel:
    """
        Base model for supervised learning.

        Arguments:
        ----------
            adj: `scipy.sparse.csr_matrix` (or `csc_matrix`) with shape (N, N)
                The input `symmetric` adjacency matrix, where `N` is the number of nodes 
                in graph.
            features: `np.array` with shape (N, F)
                The input node feature matrix, where `F` is the dimension of node features.
            labels: `np.array` with shape (N,)
                The ground-truth labels for all nodes in graph.
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

    def __init__(self, adj, features, labels, **kwargs):

        seed = kwargs.pop('seed', None)
        device = kwargs.pop('device', 'CPU:0')
        
        self.device_name = device
        self.name = kwargs.pop('name', self.__class__.__name__)
        self.n_nodes, self.n_features = features.shape
        self.n_classes = labels.max() + 1
        
        self.adjacency_matrix = adj
        self.feature_matrix = features
        self.labels = labels

        if seed:
            tf.random.set_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
        if kwargs:
            raise ValueError(f'Invalid arguments of `{list(kwargs.keys())}`.')
            
        self.seed = seed
        self.device = tf.device(device)

        self.model = None
        self.built = None
        self.index_train = None
        self.index_val = None
        self.index_test = None
        self.do_before_train = None
        self.do_before_validation = None
        self.do_before_test = None
        self.do_before_predict = None
        self.sparse = True
        self.custom_objects = None # used for save/load model

        self.log_path = f'./log/{self.name}_weights'
        
    def preprocess(self, adj, features):
        """
            Preprocess the input adjacency matrix and feature matrix, e.g., normalize.
            And convert some of necessary to tensor.
            
        Arguments:
        ----------
            adj: `scipy.sparse.csr_matrix` (or `csr_matrix`) with shape (N, N)
                The input `symmetric` adjacency matrix, where `N` is the number of nodes
                in graph.
            features: `np.array` with shape (N, F) 
                The input node feature matrix, where `F` is the dimension of node features.            
        """
        raise NotImplementedError
        

    def build(self):
        """
            Build the model using customized hyperparameters.
            
        Note:
        ----------
            This method must be called before training/testing/predicting. 
            Use `model.build()`. The following `Arguments` are only commonly used 
            hyperparameters, and other model-specific parameters are not specified.
            
            
        Arguments:
        ----------
            hidden_layers: `list` of integer 
                The number of hidden units of model. Note: the last hidden unit (`n_classes`)
                aren't nececcary to specified and it will be automatically added in the last 
                layer. 
            activations: `list` of string
                The activation function of model. Note: the last activation function (`softmax`) 
                aren't nececcary to specified and it will be automatically spefified in the 
                final output.              
            dropout: Float scalar
                Dropout rate for the hidden outputs.
            learning_rate: Float scalar
                Learning rate for the model.
            l2_norm: Float scalar
                L2 normalize for the hidden layers. (usually only used in the first layer)
            use_bias: Boolean
                Whether to use bias in the hidden layers.
            
        """                
        raise NotImplementedError

    def train(self, index_train, index_val=None,
              epochs=200, early_stopping=None, validation=True,
              verbose=False, restore_best=True, log_path=None, save_model=False,
              best_metric='val_accuracy', early_stop_metric='val_loss'):
        """
            Train the model for the input `index_train` of nodes or `sequence`.
            
        Note:
        ----------
            You must compile your model before training/testing/predicting. Use `model.build()`.
        
        Arguments:
        ----------
            index_train: `np.array`, `list
                Integer scalar or `graphgallery.NodeSequence`, the index of nodes (or sequence) 
                that will used in training.    
            index_val: `np.array`, `list`
                Integer scalar or `graphgallery.NodeSequence`, the index of nodes (or sequence) 
                that will used in validation. (default :obj: `None`, i.e., do not use validation 
                during training)
            epochs: Postive integer
                The number of epochs of training.(default :obj: `200`)
            early_stopping: Postive integer or None
                The number of early stopping patience during training. (default :obj: `None`, 
                i.e., do not use early stopping during training)
            validation: Boolean
                Whether to use validation during traiing, if `True`, the `index_val` must be 
                specified. (default :obj: `True`)
            verbose: Boolean
                Whether to show the training details. (default :obj: `None`)
            restore_best: Boolean
                Whether to restore the best result (accuracy of loss depend on `best_metrix`) 
                of training or validation (depend on `validation` is `False` or `True`). 
                (default :obj: `True`)
            log_path: String or None
                The path of saved weights/model. (default :obj: `None`, i.e., 
                `./log/{self.name}_weights`)
            save_model: Boolean
                Whether to save the whole model or weights only, if `True`, the `self.custom_objects`
                must be speficied if you are using customized `layer` or `loss` and so on.
            best_metric: String
                One of (val_loss, val_accuracy, loss, accuracy), it determines which metric will be
                used for `restore_best`. (default :obj: `val_accuracy`)
            early_stop_metric: String
                One of (val_loss, val_accuracy, loss, accuracy), it determines which metric will be 
                used for early stopping. (default :obj: `val_loss`)
                
        Return:
        ----------
            history: graphgallery.utils.History
                tensorflow like `history` instance.
        """        
        
        # Check if model has been built
        if not self.built:
            raise RuntimeError('You must compile your model before training/testing/predicting. Use `model.build()`.')
            
        if isinstance(index_train, Sequence):
            train_data = index_train
        else:
            train_data = self.train_sequence(index_train)
            self.index_train = self._check_and_convert(index_train)
            

        if validation and index_val is None:
            raise RuntimeError('`index_val` must be specified when `validation=True`.')
        
        if index_val is not None:
            if isinstance(index_val, Sequence):
                val_data = index_val
            else:
                val_data = self.test_sequence(index_val)
                self.index_val = self._check_and_convert(index_val)

        history = History(best_metric=best_metric,
                          early_stop_metric=early_stop_metric)

        if log_path is None:
            log_path = self.log_path

        if not validation:
            history.register_best_metric('accuracy')
            history.register_early_stop_metric('loss')
            
        if verbose:
            pbar = tqdm(range(1, epochs+1))
        else:
            pbar = range(1, epochs+1)
            
        for epoch in pbar:

            if self.do_before_train is not None:
                self.do_before_train()

            loss, accuracy = self.do_forward(train_data)
#             if tf.is_tensor(loss): loss = loss.numpy()
#             if tf.is_tensor(accuracy): accuracy = accuracy.numpy()
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
                self.save(log_path, save_model=save_model)

            # early stopping
            if early_stopping and history.time_to_early_stopping(early_stopping):
                msg = f'Early stopping with patience {early_stopping}.'
                if verbose:
                    pbar.set_description(msg)  
                else:
                    print(msg)
                break

            if verbose:
                msg = f'loss {loss:.2f}, accuracy {accuracy:.2%}'
                if validation:
                    msg += f', val_loss {val_loss:.2f}, val_accuracy {val_accuracy:.2%}'
                pbar.set_description(msg) 
                
        if restore_best:
            self.load(log_path, save_model=save_model)

        return history

    def test(self, index, **kwargs):
        """
            Test the output accuracy for the `index` of nodes or `sequence`.
            
        Note:
        ----------
            You must compile your model before training/testing/predicting.
            Use `model.build()`.
        
        Arguments:
        ----------
            index: `np.array`, `list`, integer scalar or `graphgallery.NodeSequence`
            The index of nodes (or sequence) that will be tested.    
            
            **kwargs (optional): Additional arguments of
                :method:`do_before_test`.   
                
        Return:
        ----------
            loss: Float scalar
                Output loss of forward propagation. 
            accuracy: Float scalar
                Output accuracy of prediction.        
        """        

        if not self.built:
            raise RuntimeError('You must compile your model before training/testing/predicting. Use `model.build()`.')
            
        if isinstance(index, Sequence):
            test_data = index
        else:
            test_data = self.test_sequence(index)
            self.index_test = self._check_and_convert(index)
        
        if self.do_before_test is not None:
            self.do_before_test(**kwargs)

        loss, accuracy = self.do_forward(test_data, training=False)

        return loss, accuracy

    def do_forward(self, sequence, training=True):
        """
            Forward propagation for the input `sequence`. This method will be called 
            in `train` and `test`, you can rewrite it for you customized training/testing 
            process. If you want to specify your customized data during traing/testing/predicting, 
            you can implement a sub-class of `graphgallery.NodeSequence`, wich is iterable 
            and yields `inputs` and `labels` in each iteration. 
            
            
        Note:
        ----------
            You must compile your model before training/testing/predicting. 
            Use `model.build()`.
        
        Arguments:
        ----------
            sequence: `graphgallery.NodeSequence`
                The input `sequence`.    
            trainng (Boolean, optional): 
                Indicating training or test procedure. (default: :obj:`True`)
                
        Return:
        ----------
            loss: Float scalar
                Output loss of forward propagation.
            accuracy: Float scalar
                Output accuracy of prediction.
        
        """        

        if training:
            forward_fn = self.model.train_on_batch
        else:
            forward_fn = self.model.test_on_batch
            
        self.model.reset_metrics()

        with self.device:
            for inputs, labels in sequence:
                loss, accuracy = forward_fn(x=inputs, y=labels, reset_metrics=False)

        return loss, accuracy

    def predict(self, index, **kwargs):
        """
            Predict the output probability for the `index` of nodes.
            
        Note:
        ----------
            You must compile your model before training/testing/predicting. 
            Use `model.build()`.
        
        Arguments:
        ----------
            index: `np.array`, `list` or integer scalar
                The index of nodes that will be computed.    
            
            **kwargs (optional): Additional arguments of
                :method:`do_before_predict`.   
                
        Return:
        ----------
            The predicted probability of each class for each node, 
            shape (len(index), n_classes).
        
        """
        
        if not self.built:
            raise RuntimeError('You must compile your model before training/testing/predicting. Use `model.build()`.')
            
        if self.do_before_predict is not None:
            self.do_before_predict(**kwargs)
            
    def train_sequence(self, index):
        """
            Construct the training sequence for the `index` of nodes.
            
        
        Arguments:
        ----------
            index: `np.array`, `list` or integer scalar
                The index of nodes used in training.
            
        Return:
        ----------
            The sequence of `graphgallery.NodeSequence` for the nodes.
        
        """        
        
        raise NotImplementedError
        
    def test_sequence(self, index):
        """
            Construct the testing sequence for the `index` of nodes.
            
        Note:
        ----------
            If not implemented, this method will call `train_sequence` automatically.
        
        Arguments:
        ----------
            index: `np.array`, `list` or integer scalar
                The index of nodes used in testing.
            
        Return:
        ----------
            The sequence of `graphgallery.NodeSequence` for the nodes.
        """
        return self.train_sequence(index) 
    
    def test_predict(self, index):
        """
            Predict the output accuracy for the `index` of nodes.
            
        Note:
        ----------
            You must compile your model before training/testing/predicting. 
            Use `model.build()`.
        
        Arguments:
        ----------
            index: `np.array`, `list` or integer scalar
                The index of nodes that will be computed.    
            
        Return:
        ----------
            accuracy: Float scalar
                The output accuracy of the `index` of nodes.
        
        """        
        index = self._check_and_convert(index)
        logit = self.predict(index)
        predict_class = logit.argmax(1)
        labels = self.labels[index]
        return (predict_class == labels).mean()
    
    @tf.function
    def __call__(self, inputs):
        return self.model(inputs)

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
        if not isinstance(features, np.ndarray):
            raise TypeError('feature matrix must be the instance of np.array.')
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
        if tf.is_tensor(index):
            return tf.cast(index, tf.int32)
        
        if isinstance(index, int):
            index = np.asarray([index])
        elif isinstance(index, list):
            index = np.asarray(index)
        elif isinstance(index, np.ndarray):
            pass
        else:
            raise TypeError('`index` should be either list, int or np.ndarray!')
        return index.astype('int32')

    def save(self, path=None, save_model=False):
        if not os.path.exists('log'):
            os.makedirs('log')       
            print('mkdir /log')
            
        if path is None:
            path = self.log_path
            
        if save_model:
            self.model.save(self.log_path, save_format='h5')
        else:
            self.model.save_weights(path)

    def load(self, path=None, save_model=False):
        if path is None:
            path = self.log_path
            
        if save_model:
            self.model = tf.keras.models.load_model(path, custom_objects=self.custom_objects)
        else:
            self.model.load_weights(path)
            
    @property
    def weights(self):
        """
            Return the weights of model, type `tf.Tensor`.
        """
        return self.model.weights

    @property
    def np_weights(self):
        """
            Return the weights of model, type `np.array`.
        """        
        return [weight.numpy() for weight in self.weights]
    
    @property
    def trainable_variables(self):
        """
            Return the trainable weights of model, type `tf.Tensor`.
        """        
        return self.model.trainable_variables

    @property
    def close(self):
        """
            Close the session of model and set `built` to False.
        """
        # del self.model
        self.built = None
        K.clear_session()
        
    def __repr__(self):
        return self.name + ' in ' + self.device_name

        
        
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

class UnsupervisedModel:
    """
        Base model for unsupervised learning.

        Arguments:
        ----------
            adj: `scipy.sparse.csr_matrix` (or `csc_matrix`) with shape (N, N)
                The input `symmetric` adjacency matrix, where `N` is the number of nodes 
                in graph.
            features: `np.array` with shape (N, F)
                The input node feature matrix, where `F` is the dimension of node features.
            labels: `np.array` with shape (N,)
                The ground-truth labels for all nodes in graph.
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

    def __init__(self, adj, features, labels, **kwargs):

        seed = kwargs.pop('seed', None)
        device = kwargs.pop('device', 'CPU:0')
        
        self.device_name = device        
        self.name = kwargs.pop('name', self.__class__.__name__)
        self.n_nodes, self.n_features = features.shape
        self.n_classes = labels.max() + 1
        self.adj, self.features = adj, features
        self.labels = labels

        if seed:
            tf.random.set_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
        if kwargs:
            raise ValueError(f'Invalid arguments of `{list(kwargs.keys())}`.')
            
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
        return index.astype('int32')      

    def __repr__(self):
        return self.name + ' in ' + self.device_name
