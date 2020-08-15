import os
import logging
import warnings
import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.python.keras import callbacks as callbacks_module
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import History as tf_History

from graphgallery.nn.models import BaseModel
from graphgallery.utils.history import History
from graphgallery.utils.tqdm import tqdm
from graphgallery import asintarr, Bunch

# Ignora warnings:
#     UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
#     This is caused by `tf.gather` and it will be solved in future tensorflow version.

warnings.filterwarnings(
    'ignore', '.*Converting sparse IndexedSlices to a dense Tensor of unknown shape.*')


class SemiSupervisedModel(BaseModel):
    """
        Base model for semi-supervised learning.

        Arguments:
        ----------
            adj: shape (N, N), Scipy sparse matrix if  `is_adj_sparse=True`, 
                Numpy array-like (or matrix) if `is_adj_sparse=False`.
                The input `symmetric` adjacency matrix, where `N` is the number 
                of nodes in graph.
            features: Numpy array-like with shape (N, F)
                The input node feature matrix, where `F` is the dimension of node features.
            labels: Numpy array-like with shape (N,)
                The ground-truth labels for all nodes in graph.
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

    def __init__(self, adj, x, labels=None, device='CPU:0', seed=None, name=None, **kwargs):
        super().__init__(adj, x, labels, device, seed, name, **kwargs)

    def preprocess(self, adj, x):
        """Preprocess the input adjacency matrix and feature matrix, e.g., normalization.
        And convert them to tf.tensor. 

        Arguments:
        ----------
            adj: shape (N, N), `scipy.sparse.csr_matrix` (or `csc_matrix`) if 
                `is_adj_sparse=True`, Numpy array-like if `is_adj_sparse=False`.
                The input `symmetric` adjacency matrix, where `N` is the number 
                of nodes in graph.
            x: shape (N, F), `scipy.sparse.csr_matrix` (or `csc_matrix`) if 
                `is_x_sparse=True`, Numpy array-like if `is_x_sparse=False`.
                The input node feature matrix, where `F` is the dimension of node features.

        Note:
        ----------
            By default, `adj` is sparse matrix and `x` is dense array. Both of them are 
            2-D matrices.
        """
        # check the input adj and x, and convert them to appropriate forms
        self.adj, self.x = self._check_inputs(adj, x)
        self.n_nodes, self.n_features = x.shape

    def build(self):
        """
            Build the model using customized hyperparameters.

        Note:
        ----------
            This method must be called before training/testing/predicting. 
            Use `model.build()`. The following `Arguments` are only commonly used 
            arguments, and other model-specific arguments are not introduced as follows.


        Arguments:
        ----------
            hiddens: `list` of integer or integer scalar 
                The number of hidden units of model. Note: the last hidden unit (`n_classes`)
                aren't nececcary to specified and it will be automatically added in the last 
                layer. 
            activations: `list` of string or string
                The activation function of model. Note: the last activation function (`softmax`) 
                aren't nececcary to specified and it will be automatically spefified in the 
                final output.              
            dropouts: `list` of float scalar or float scalar
                Dropout rates for the hidden outputs.
            l2_norms:  `list` of float scalar or float scalar
                L2 normalize parameters for the hidden layers. (only used in the hidden layers)
            lr: Float scalar
                Learning rate for the training model.
            use_bias: Boolean
                Whether to use bias in the hidden layers.

        """
        raise NotImplementedError

    def train_v1(self, idx_train, idx_val=None,
              epochs=200, early_stopping=None,
              verbose=False, save_best=True, weight_path=None, as_model=False,
              monitor='val_acc', early_stop_metric='val_loss'):
        """
            Train the model for the input `idx_train` of nodes or `sequence`.

        Note:
        ----------
            You must compile your model before training/testing/predicting. Use `model.build()`.

        Arguments:
        ----------
            idx_train: Numpy array-like, `list`, Integer scalar or `graphgallery.NodeSequence`
                the index of nodes (or sequence) that will be used during training.    
            idx_val: Numpy array-like, `list`, Integer scalar or `graphgallery.NodeSequence`, optional
                the index of nodes (or sequence) that will be used for validation. 
                (default :obj: `None`, i.e., do not use validation during training)
            epochs: Postive integer
                The number of epochs of training.(default :obj: `200`)
            early_stopping: Postive integer or None
                The number of early stopping patience during training. (default :obj: `None`, 
                i.e., do not use early stopping during training)
            verbose: Boolean
                Whether to show the training details. (default :obj: `None`)
            save_best: Boolean
                Whether to save the best weights (accuracy of loss depend on `monitor`) 
                of training or validation (depend on `validation` is `False` or `True`). 
                (default :bool: `True`)
            weight_path: String or None
                The path of saved weights/model. (default :obj: `None`, i.e., 
                `./log/{self.name}_weights`)
            as_model: Boolean
                Whether to save the whole model or weights only, if `True`, the `self.custom_objects`
                must be speficied if you are using customized `layer` or `loss` and so on.
            monitor: String
                One of (val_loss, val_acc, loss, acc), it determines which metric will be
                used for `save_best`. (default :obj: `val_acc`)
            early_stop_metric: String
                One of (val_loss, val_acc, loss, acc), it determines which metric will be 
                used for early stopping. (default :obj: `val_loss`)

        Return:
        ----------
            history: graphgallery.utils.History
                tensorflow like `history` instance.
        """

        # TODO use tensorflow callbacks

        ############# Record paras ###########
        local_paras = locals()
        local_paras.pop('self')
        local_paras.pop('idx_train')
        local_paras.pop('idx_val')
        paras = Bunch(**local_paras)
        ######################################

        # Check if model has been built
        if self.model is None:
            raise RuntimeError('You must compile your model before training/testing/predicting. Use `model.build()`.')

        if isinstance(idx_train, Sequence):
            train_data = idx_train
        else:
            idx_train = asintarr(idx_train)
            train_data = self.train_sequence(idx_train)
            self.idx_train = idx_train

        validation = idx_val is not None

        if validation:
            if isinstance(idx_val, Sequence):
                val_data = idx_val
            else:
                idx_val = asintarr(idx_val)
                val_data = self.test_sequence(idx_val)
                self.idx_val = idx_val
        else:
            monitor = 'acc' if monitor[:3] == 'val' else monitor

        history = History(monitor_metric=monitor,
                          early_stop_metric=early_stop_metric)

        if not weight_path:
            weight_path = self.weight_path
        
        if not weight_path.endswith('.h5'):
            weight_path += '.h5'

        ############# Record paras ###########
        paras.update(Bunch(weight_path=weight_path))
        # update all parameters
        self.paras.update(paras)
        self.train_paras.update(paras)
        ######################################

        if validation is None:
            history.register_monitor_metric('acc')
            history.register_early_stop_metric('loss')

        if verbose:
            pbar = tqdm(range(1, epochs+1))
        else:
            pbar = range(1, epochs+1)

        for epoch in pbar:

            if self.do_before_train:
                self.do_before_train()

            loss, accuracy = self.do_forward(train_data)
            train_data.on_epoch_end()

            history.add_results(loss, 'loss')
            history.add_results(accuracy, 'acc')

            if validation:
                if self.do_before_validation:
                    self.do_before_validation()

                val_loss, val_accuracy = self.do_forward(val_data, training=False)

                history.add_results(val_loss, 'val_loss')
                history.add_results(val_accuracy, 'val_acc')

            # record eoch and running times
            history.record_epoch(epoch)

            if save_best and history.save_best:
                self.save(weight_path, as_model=as_model)

            # early stopping
            if early_stopping and history.time_to_early_stopping(early_stopping):
                msg = f'Early stopping with patience {early_stopping}.'
                if verbose:
                    pbar.set_description(msg)
                    pbar.close()
                break

            if verbose:
                msg = f'loss {loss:.2f}, acc {accuracy:.2%}'
                if validation:
                    msg += f', val_loss {val_loss:.2f}, val_acc {val_accuracy:.2%}'
                pbar.set_description(msg)

        if save_best:
            self.load(weight_path, as_model=as_model)
            os.remove(weight_path)

        return history

    def train(self, idx_train, idx_val=None,
                 epochs=200, early_stopping=None,
                 verbose=False, save_best=True, weight_path=None, as_model=False,
                 monitor='val_acc', early_stop_metric='val_loss', callbacks=None, **kwargs):
        """
            Train the model for the input `idx_train` of nodes or `sequence`.

        Note:
        ----------
            You must compile your model before training/testing/predicting. Use `model.build()`.

        Arguments:
        ----------
            idx_train: Numpy array-like, `list`, Integer scalar or `graphgallery.NodeSequence`
                the index of nodes (or sequence) that will be used during training.    
            idx_val: Numpy array-like, `list`, Integer scalar or `graphgallery.NodeSequence`, optional
                the index of nodes (or sequence) that will be used for validation. 
                (default :obj: `None`, i.e., do not use validation during training)
            epochs: Postive integer
                The number of epochs of training.(default :obj: `200`)
            early_stopping: Postive integer or None
                The number of early stopping patience during training. (default :obj: `None`, 
                i.e., do not use early stopping during training)
            verbose: Boolean
                Whether to show the training details. (default :obj: `None`)
            save_best: Boolean
                Whether to save the best weights (accuracy of loss depend on `monitor`) 
                of training or validation (depend on `validation` is `False` or `True`). 
                (default :bool: `True`)
            weight_path: String or None
                The path of saved weights/model. (default :obj: `None`, i.e., 
                `./log/{self.name}_weights`)
            as_model: Boolean
                Whether to save the whole model or weights only, if `True`, the `self.custom_objects`
                must be speficied if you are using customized `layer` or `loss` and so on.
            monitor: String
                One of (val_loss, val_acc, loss, acc), it determines which metric will be
                used for `save_best`. (default :obj: `val_acc`)
            early_stop_metric: String
                One of (val_loss, val_acc, loss, acc), it determines which metric will be 
                used for early stopping. (default :obj: `val_loss`)
            callbacks: tensorflow.keras.callbacks. (default :obj: `None`)
            kwargs: other keyword arguments.

        Return:
        ----------
            A `tf.keras.callbacks.History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).

        """
        ############# Record paras ###########
        local_paras = locals()
        local_paras.pop('self')
        local_paras.pop('idx_train')
        local_paras.pop('idx_val')
        paras = Bunch(**local_paras)
        ######################################
        model = self.model
        model.stop_training = False
        
        # Check if model has been built
        if model is None:
            raise RuntimeError('You must compile your model before training/testing/predicting. Use `model.build()`.')

        if isinstance(idx_train, Sequence):
            train_data = idx_train
        else:
            idx_train = asintarr(idx_train)
            train_data = self.train_sequence(idx_train)
            self.idx_train = idx_train

        validation = idx_val is not None

        if validation:
            if isinstance(idx_val, Sequence):
                val_data = idx_val
            else:
                idx_val = asintarr(idx_val)
                val_data = self.test_sequence(idx_val)
                self.idx_val = idx_val
        else:
            monitor = 'acc' if monitor[:3] == 'val' else monitor



        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(callbacks)
            
        his = tf_History()
        callbacks.append(his) 
        
        if early_stopping:
            es_callback = EarlyStopping(monitor=early_stop_metric,
                                        patience=early_stopping,
                                        mode='auto',
                                        verbose=kwargs.pop('es_verbose', 1))
            callbacks.append(es_callback)

        if save_best:
            if not weight_path:
                if not os.path.exists(self.weight_dir):
                    os.makedirs(self.weight_dir)
                    logging.log(logging.WARNING, f"Make Directory in {self.weight_dir}")
                    
                weight_path = self.weight_path

            if not weight_path.endswith('.h5'):
                weight_path += '.h5'
                

            mc_callback = ModelCheckpoint(weight_path,
                                          monitor=monitor,
                                          save_best_only=True,
                                          save_weights_only=not as_model,
                                          verbose=0)
            callbacks.append(mc_callback)
        callbacks.set_model(model)

        ############# Record paras ###########
        paras.update(Bunch(weight_path=weight_path))
        # update all parameters
        self.paras.update(paras)
        self.train_paras.update(paras)
        ######################################

        # leave it blank for the future
        allowed_kwargs = set([])
        unknown_kwargs = set(kwargs.keys()) - allowed_kwargs
        if unknown_kwargs:
            raise TypeError(
                "Invalid keyword argument(s) in `__init__`: %s" % (unknown_kwargs,))

        callbacks.on_train_begin()
        
        if verbose:
            pbar = tqdm(range(1, epochs+1))
        else:
            pbar = range(1, epochs+1)
            
        for epoch in pbar:
            callbacks.on_epoch_begin(epoch)

            if self.do_before_train:
                self.do_before_train()

            callbacks.on_train_batch_begin(0)
            loss, accuracy = self.do_forward(train_data)
            train_data.on_epoch_end()

            training_logs = {'loss': loss, 'acc': accuracy}
            callbacks.on_train_batch_end(0, training_logs)

            if validation:
                if self.do_before_validation:
                    self.do_before_validation()

                val_loss, val_accuracy = self.do_forward(val_data, training=False)
                training_logs.update({'val_loss': val_loss, 'val_acc': val_accuracy})

            callbacks.on_epoch_end(epoch, training_logs)
            
            if verbose:
                msg = "<"
                for key, val in training_logs.items():
                    msg += f"{key.title()} = {val:.4f} "
                msg += ">"
                pbar.set_description(msg)            

            if model.stop_training:
                break

        callbacks.on_train_end()

        if save_best:
            self.load(weight_path, as_model=as_model)
            if os.path.exists(weight_path):
                os.remove(weight_path)

        return his

    def train_v2(self, idx_train, idx_val=None,
                 epochs=200, early_stopping=None,
                 verbose=False, save_best=True, weight_path=None, as_model=False,
                 monitor='val_acc', early_stop_metric='val_loss', callbacks=None, **kwargs):
        """
            Train the model for the input `idx_train` of nodes or `sequence`.

        Note:
        ----------
            You must compile your model before training/testing/predicting. Use `model.build()`.

        Arguments:
        ----------
            idx_train: Numpy array-like, `list`, Integer scalar or `graphgallery.NodeSequence`
                the index of nodes (or sequence) that will be used during training.    
            idx_val: Numpy array-like, `list`, Integer scalar or `graphgallery.NodeSequence`, optional
                the index of nodes (or sequence) that will be used for validation. 
                (default :obj: `None`, i.e., do not use validation during training)
            epochs: Postive integer
                The number of epochs of training.(default :obj: `200`)
            early_stopping: Postive integer or None
                The number of early stopping patience during training. (default :obj: `None`, 
                i.e., do not use early stopping during training)
            verbose: Boolean
                Whether to show the training details. (default :obj: `None`)
            save_best: Boolean
                Whether to save the best weights (accuracy of loss depend on `monitor`) 
                of training or validation (depend on `validation` is `False` or `True`). 
                (default :bool: `True`)
            weight_path: String or None
                The path of saved weights/model. (default :obj: `None`, i.e., 
                `./log/{self.name}_weights`)
            as_model: Boolean
                Whether to save the whole model or weights only, if `True`, the `self.custom_objects`
                must be speficied if you are using customized `layer` or `loss` and so on.
            monitor: String
                One of (val_loss, val_acc, loss, acc), it determines which metric will be
                used for `save_best`. (default :obj: `val_acc`)
            early_stop_metric: String
                One of (val_loss, val_acc, loss, acc), it determines which metric will be 
                used for early stopping. (default :obj: `val_loss`)
            callbacks: tensorflow.keras.callbacks. (default :obj: `None`)
            kwargs: other keyword arguments.

        Return:
        ----------
            A `tf.keras.callbacks.History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).

        """
        ############# Record paras ###########
        local_paras = locals()
        local_paras.pop('self')
        local_paras.pop('idx_train')
        local_paras.pop('idx_val')
        paras = Bunch(**local_paras)
        ######################################

        if not tf.__version__ >= '2.2.0':
            raise RuntimeError(f'This method is only work for tensorflow version >= 2.2.0.')

        # Check if model has been built
        if self.model is None:
            raise RuntimeError('You must compile your model before training/testing/predicting. Use `model.build()`.')

        if isinstance(idx_train, Sequence):
            train_data = idx_train
        else:
            idx_train = asintarr(idx_train)
            train_data = self.train_sequence(idx_train)
            self.idx_train = idx_train

        validation = idx_val is not None

        if validation:
            if isinstance(idx_val, Sequence):
                val_data = idx_val
            else:
                idx_val = asintarr(idx_val)
                val_data = self.test_sequence(idx_val)
                self.idx_val = idx_val
        else:
            monitor = 'acc' if monitor[:3] == 'val' else monitor

        model = self.model
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(callbacks,
                                                      add_history=True,
                                                      add_progbar=True,
                                                      verbose=verbose,
                                                      epochs=epochs)
        if early_stopping:
            es_callback = EarlyStopping(monitor=early_stop_metric,
                                        patience=early_stopping,
                                        mode='auto',
                                        verbose=kwargs.pop('es_verbose', 0))
            callbacks.append(es_callback)

        if save_best:
            if not weight_path:
                if not os.path.exists(self.weight_dir):
                    os.makedirs(self.weight_dir)
                    logging.log(logging.WARNING, f"Make Directory in {self.weight_dir}")                
                weight_path = self.weight_path

            if not weight_path.endswith('.h5'):
                weight_path += '.h5'

            mc_callback = ModelCheckpoint(weight_path,
                                          monitor=monitor,
                                          save_best_only=True,
                                          save_weights_only=not as_model,
                                          verbose=0)
            callbacks.append(mc_callback)
        callbacks.set_model(model)

        ############# Record paras ###########
        paras.update(Bunch(weight_path=weight_path))
        # update all parameters
        self.paras.update(paras)
        self.train_paras.update(paras)
        ######################################

        # leave it blank for the future
        allowed_kwargs = set([])
        unknown_kwargs = set(kwargs.keys()) - allowed_kwargs
        if unknown_kwargs:
            raise TypeError(
                "Invalid keyword argument(s) in `__init__`: %s" % (unknown_kwargs,))

        callbacks.on_train_begin()

        for epoch in range(epochs):
            callbacks.on_epoch_begin(epoch)

            if self.do_before_train:
                self.do_before_train()

            callbacks.on_train_batch_begin(0)
            loss, accuracy = self.do_forward(train_data)
            train_data.on_epoch_end()

            training_logs = {'loss': loss, 'acc': accuracy}
            callbacks.on_train_batch_end(0, training_logs)

            if validation:
                if self.do_before_validation:
                    self.do_before_validation()

                val_loss, val_accuracy = self.do_forward(val_data, training=False)
                training_logs.update({'val_loss': val_loss, 'val_acc': val_accuracy})

            callbacks.on_epoch_end(epoch, training_logs)

            if model.stop_training:
                break

        callbacks.on_train_end()

        if save_best:
            self.load(weight_path, as_model=as_model)
            if os.path.exists(weight_path):
                os.remove(weight_path)

        return model.history
    
    def test(self, index, **kwargs):
        """
            Test the output accuracy for the `index` of nodes or `sequence`.

        Note:
        ----------
            You must compile your model before training/testing/predicting.
            Use `model.build()`.

        Arguments:
        ----------
            index: Numpy array-like, `list`, Integer scalar or `graphgallery.NodeSequence`
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

        # TODO record test logs like self.train() or self.train_v2()
        if not self.model:
            raise RuntimeError('You must compile your model before training/testing/predicting. Use `model.build()`.')

        if isinstance(index, Sequence):
            test_data = index
        else:
            index = asintarr(index)
            test_data = self.test_sequence(index)
            self.idx_test = index

        if self.do_before_test:
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
        model = self.model

        if training:
            forward_fn = model.train_on_batch
        else:
            forward_fn = model.test_on_batch

        model.reset_metrics()

        with tf.device(self.device):
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
            index: Numpy array-like, `list` or integer scalar
                The index of nodes that will be computed.    

            **kwargs (optional): Additional arguments of
                :method:`do_before_predict`.   

        Return:
        ----------
            The predicted probability of each class for each node, 
            shape (len(index), n_classes).

        """

        if not self.model:
            raise RuntimeError('You must compile your model before training/testing/predicting. Use `model.build()`.')

        if self.do_before_predict:
            self.do_before_predict(**kwargs)

    def train_sequence(self, index):
        """
            Construct the training sequence for the `index` of nodes.


        Arguments:
        ----------
            index: Numpy array-like, `list` or integer scalar
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
            index: Numpy array-like, `list` or integer scalar
                The index of nodes used in testing.

        Return:
        ----------
            The sequence of `graphgallery.NodeSequence` for the nodes.
        """
        return self.train_sequence(index)

    def _test_predict(self, index):
        index = asintarr(index)
        logit = self.predict(index)
        predict_class = logit.argmax(1)
        labels = self.labels[index]
        return (predict_class == labels).mean()

    def __call__(self, inputs):
        return self.model(inputs)

#     @property
#     def weights(self):
#         """Return the weights of model, type `tf.Tensor`."""
#         return self.model.weights

#     def get_weights(self):
#         """Return the weights of model, type Numpy array-like."""
#         return self.model.get_weights()

#     @property
#     def trainable_variables(self):
#         """Return the trainable weights of model, type `tf.Tensor`."""
#         return self.model.trainable_variables

    def reset_weights(self):
        """reset the model to the first time.
        """
        model = self.model
        if self.backup is None:
            raise RuntimeError("You must store the `backup` before `reset_weights`."
                               "`backup` will be automatically stored when the model is built.")
        for w, wb in zip(model.weights, self.backup):
            w.assign(wb)

    def reset_optimizer(self):

        model = self.model
        if hasattr(model, 'optimizer'):
            for var in model.optimizer.variables():
                var.assign(tf.zeros_like(var))

    def reset_lr(self, value):
        model = self.model
        if not hasattr(model, 'optimizer'):
            raise RuntimeError("The model has not attribute `optimizer`!")
        model.optimizer.learning_rate.assign(value)

    @property
    def close(self):
        """Close the session of model and set `built` to False."""
        self.model = None
        K.clear_session()

    def __repr__(self):
        return 'Semi-Supervised model: ' + self.name + ' in ' + self.device
