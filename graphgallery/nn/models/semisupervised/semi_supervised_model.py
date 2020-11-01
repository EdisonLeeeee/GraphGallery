import os
import time
import copy
import logging
import warnings
import os.path as osp
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from functools import partial

from tensorflow.keras.utils import Sequence
from tensorflow.python.keras import callbacks as callbacks_module
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import History
from tensorflow.python.keras.utils.generic_utils import Progbar

import graphgallery as gg
from graphgallery.nn.models import BaseModel
from graphgallery.nn.models import training
from graphgallery.nn.functions import softmax
from graphgallery.data.io import makedirs_from_filename
from graphgallery.data import BaseGraph
from graphgallery.transforms import asintarr
from graphgallery.utils.raise_error import raise_if_kwargs



# Ignora warnings:
#     UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
#     This is caused by `tf.gather` and it will be solved in future tensorflow version.
warnings.filterwarnings(
    'ignore', '.*Converting sparse IndexedSlices to a dense Tensor of unknown shape.*')
    
class SemiSupervisedModel(BaseModel):
    def __init__(self, *graph, device='cpu:0', seed=None, name=None, **kwargs):
        super().__init__(*graph, device=device, seed=seed, name=name, **kwargs)
    
        if self.backend == "tensorflow":
            self.train_step_fn = partial(training.train_step_tf, device=self.device)
            self.test_step_fn = partial(training.test_step_tf, device=self.device)
            self.predict_step_fn = partial(training.predict_step_tf, device=self.device)
        else:
            self.train_step_fn = training.train_step_torch
            self.test_step_fn = training.test_step_torch
            self.predict_step_fn = training.predict_step_torch
        
    def process(self, *graph, **kwargs):
        """pre-process for the input graph, including manipulations
        on adjacency matrix and attribute matrix, and finally convert
        them into tensor (optional).

        Note:
        ----------
        This method will call the method 'process_step'
            and it must be implemented for processing the graph.

        Parameters:
        ----------
        graph: An instance of `graphgallery.data.Graph` or a tuple (list) of inputs.
            A sparse, attributed, labeled graph.
        kwargs: other custom keyword parameters.

        """
        if len(graph) > 0:
            if len(graph) == 1:
                graph, = graph
                if isinstance(graph, BaseGraph):
                    self.graph = graph
                elif isinstance(graph, dict):
                    self.graph.set_inputs(**graph)
            else:
                self.graph.set_inputs(*graph)
                
        return self.process_step()

    def process_step(self):
        raise NotImplementedError

    def build(self):
        """Build the model using custom hyperparameters.

        Note:
        ----------
        This method must be called before training/testing/predicting.
        Use `model.build()`. The following `Parameters` are only commonly used
        Parameters, and other model-specific Parameters are not introduced as follows.

        Parameters:
        ----------
            hiddens: `list` of integer or integer scalar
                The number of hidden units of model. Note: the last hidden unit (`n_classes`)
                aren't necessary to specified and it will be automatically added in the last
                layer.
            activations: `list` of string or string
                The activation function of model. Note: the last activation function (`softmax`)
                aren't necessary to specified and it will be automatically specified in the
                final output.
            dropout: float scalar
                Dropout rate for the hidden outputs.
            l2_norm:  float scalar
                L2 normalize parameters for the hidden layers. (only used in the hidden layers)
            lr: float scalar
                Learning rate for the training model.
            use_bias: bool
                Whether to use bias in the hidden layers.

        """
        raise NotImplementedError

    def build_from_model(self, model):
        """Build the model using custom model.

        Note:
        ----------
        This method must be called before training/testing/predicting.
            Use `model.build_from_model(model)` where the input `model` is
            a TensorFlow model or PyTorch Model.

        Parameters:
        ----------
        model: a TensorFlow model or PyTorch Model

        """
        # TODO: check for the input model
        if self.backend == "tensorflow":
            with tf.device(self.device):
                self.model = model
        else:
            self.model = model.to(self.device)

    def train(self, idx_train, idx_val=None,
              epochs=200, early_stopping=None,
              verbose=0, save_best=True, weight_path=None, as_model=False,
              monitor='val_acc', early_stop_metric='val_loss', callbacks=None, **kwargs):
        """Train the model for the input `idx_train` of nodes or `sequence`.

        Note:
        ----------
        You must compile your model before training/testing/predicting. Use `model.build()`.

        Parameters:
        ----------
        idx_train: Numpy array-like, `list`, Integer scalar or `graphgallery.Sequence`
            The index of nodes (or sequence) that will be used during training.
        idx_val: Numpy array-like, `list`, Integer scalar or
            `graphgallery.Sequence`, optional
            The index of nodes (or sequence) that will be used for validation.
            (default :obj: `None`, i.e., do not use validation during training)
        epochs: Positive integer
            The number of epochs of training.(default :obj: `200`)
        early_stopping: Positive integer or None
            The number of early stopping patience during training. (default :obj: `None`,
            i.e., do not use early stopping during training)
        verbose: int in {0, 1, 2, 3, 4}
                'verbose=0': not verbose; 
                'verbose=1': Progbar (one line, detailed); 
                'verbose=2': Progbar (one line, omitted); 
                'verbose=3': Progbar (multi line, detailed); 
                'verbose=4': Progbar (multi line, omitted); 
            (default :obj: 0)
        save_best: bool
            Whether to save the best weights (accuracy of loss depend on `monitor`)
            of training or validation (depend on `validation` is `False` or `True`).
            (default :bool: `True`)
        weight_path: String or None
            The path of saved weights/model. (default :obj: `None`, i.e.,
            `./log/{self.name}_weights`)
        as_model: bool
            Whether to save the whole model or weights only, if `True`, the `self.custom_objects`
            must be speficied if you are using custom `layer` or `loss` and so on.
        monitor: String
            One of (val_loss, val_acc, loss, acc), it determines which metric will be
            used for `save_best`. (default :obj: `val_acc`)
        early_stop_metric: String
            One of (val_loss, val_acc, loss, acc), it determines which metric will be
            used for early stopping. (default :obj: `val_loss`)
        callbacks: tensorflow.keras.callbacks. (default :obj: `None`)
        kwargs: other keyword Parameters.

        Return:
        ----------
        A `tf.keras.callbacks.History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).

        """
        raise_if_kwargs(kwargs)
        if not (isinstance(verbose, int) and 0<=verbose<=4):
            raise ValueError("'verbose=0': not verbose"
                             "'verbose=1': Progbar(one line, detailed), "
                             "'verbose=2': Progbar(one line, omitted), "
                             "'verbose=3': Progbar(multi line, detailed), "
                             "'verbose=4': Progbar(multi line, omitted), "
                             f"but got {verbose}")
        model = self.model
        # Check if model has been built
        if model is None:
            raise RuntimeError(
                'You must compile your model before training/testing/predicting. Use `model.build()`.')

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

        history = History()
        callbacks.append(history)
        
        if early_stopping:
            es_callback = EarlyStopping(monitor=early_stop_metric,
                                        patience=early_stopping,
                                        mode='auto',
                                        verbose=kwargs.pop('es_verbose', 1))
            callbacks.append(es_callback)

        if save_best:
            if not weight_path:
                weight_path = self.weight_path
            else:
                self.weight_path = weight_path

            makedirs_from_filename(weight_path)

            if not weight_path.endswith(gg.file_postfix()):
                weight_path = weight_path + gg.file_postfix()

            mc_callback = ModelCheckpoint(weight_path,
                                          monitor=monitor,
                                          save_best_only=True,
                                          save_weights_only=not as_model,
                                          verbose=0)
            callbacks.append(mc_callback)
            
        callbacks.set_model(model)
        model.stop_training = False
        callbacks.on_train_begin()

        if verbose:
            stateful_metrics = {"acc", 'loss', 'val_acc', 'val_loss', 'time'}
            if verbose <=2:
                progbar = Progbar(target=epochs, verbose=verbose, stateful_metrics=stateful_metrics)
            print("Training...")

        begin_time = time.perf_counter()
        try:
            for epoch in range(epochs):
                if verbose > 2:
                    progbar = Progbar(target=len(train_data), verbose=verbose - 2, stateful_metrics=stateful_metrics)

                callbacks.on_epoch_begin(epoch)
                callbacks.on_train_batch_begin(0)
                loss, accuracy = self.train_step(train_data)

                training_logs = {'loss': loss, 'acc': accuracy}
                if validation:
                    val_loss, val_accuracy = self.test_step(val_data)
                    training_logs.update(
                        {'val_loss': val_loss, 'val_acc': val_accuracy})
                    val_data.on_epoch_end()

                callbacks.on_train_batch_end(len(train_data), training_logs)
                callbacks.on_epoch_end(epoch, training_logs)

                train_data.on_epoch_end()

                if verbose:
                    time_passed = time.perf_counter() - begin_time
                    training_logs.update({'time': time_passed})                
                    if verbose > 2:
                        print(f"Epoch {epoch+1}/{epochs}")
                        progbar.update(len(train_data), training_logs.items())
                    else:
                        progbar.update(epoch + 1, training_logs.items())


                if model.stop_training:
                    break
                    
        finally:
            callbacks.on_train_end()
            # to avoid unexpected termination of the model
            if save_best:
                self.load(weight_path, as_model=as_model)
                self.remove_weights()

        return history

    def test(self, index, verbose=1):
        """
            Test the output accuracy for the `index` of nodes or `sequence`.

        Note:
        ----------
        You must compile your model before training/testing/predicting.
        Use `model.build()`.

        Parameters:
        ----------
        index: Numpy array-like, `list`, Integer scalar or `graphgallery.Sequence`
            The index of nodes (or sequence) that will be tested.


        Return:
        ----------
        loss: Float scalar
            Output loss of forward propagation.
        accuracy: Float scalar
            Output accuracy of prediction.
        """

        if not self.model:
            raise RuntimeError(
                'You must compile your model before training/testing/predicting. Use `model.build()`.')

        if isinstance(index, Sequence):
            test_data = index
        else:
            index = asintarr(index)
            test_data = self.test_sequence(index)
            self.idx_test = index

        if verbose:
            print("Testing...")
            
        stateful_metrics = {"test_acc", 'test_loss', 'time'}
        progbar = Progbar(target=len(test_data), verbose=verbose, stateful_metrics=stateful_metrics)
        begin_time = time.perf_counter()
        loss, accuracy = self.test_step(test_data)
        time_passed = time.perf_counter() - begin_time
        progbar.update(len(test_data), [('test_loss', loss), ('test_acc', accuracy), ('time', time_passed)])
        return loss, accuracy

    def train_step(self, sequence):
        """
        Forward propagation for the input `sequence`. This method will be called
        in `train`. If you want to specify your custom data during training/testing/predicting,
        you can implement a subclass of `graphgallery.Sequence`, which is iterable
        and yields `inputs` and `labels` in each iteration.


        Note:
        ----------
        You must compile your model before training/testing/predicting.
            Use `model.build()`.

        Parameters:
        ----------
        sequence: `graphgallery.Sequence`
            The input `sequence`.

        Return:
        ----------
        loss: Float scalar
            Output loss of forward propagation.
        accuracy: Float scalar
            Output accuracy of prediction.

        """
        return self.train_step_fn(self.model, sequence)

    def test_step(self, sequence):
        """
        Forward propagation for the input `sequence`. This method will be called
        in `test`. If you want to specify your custom data during training/testing/predicting,
        you can implement a subclass of `graphgallery.Sequence`, which is iterable
        and yields `inputs` and `labels` in each iteration.

        Note:
        ----------
        You must compile your model before training/testing/predicting.
            Use `model.build()`.

        Parameters:
        ----------
        sequence: `graphgallery.Sequence`
            The input `sequence`.

        Return:
        ----------
        loss: Float scalar
            Output loss of forward propagation.
        accuracy: Float scalar
            Output accuracy of prediction.

        """
        return self.test_step_fn(self.model, sequence)


    def predict(self, index=None, return_prob=True):
        """
        Predict the output probability for the input node index.


        Note:
        ----------
        You must compile your model before training/testing/predicting.
            Use `model.build()`.

        Parameters:
        ----------
        index: Numpy 1D array, optional.
            The indices of nodes to predict.
            if None, predict the all nodes.

        return_prob: bool.
            whether to return the probability of prediction.

        Return:
        ----------
        The predicted probability of each class for each node,
            shape (n_nodes, n_classes).

        """

        if not self.model:
            raise RuntimeError(
                'You must compile your model before training/testing/predicting. Use `model.build()`.')

        if index is None:
            index = np.arange(self.graph.n_nodes, dtype=gg.intx())
        else:
            index = asintarr(index)
        sequence = self.predict_sequence(index)
        logit = self.predict_step(sequence)
        if return_prob:
            logit = softmax(logit)
        return logit

    def predict_step(self, sequence):
        return self.predict_step_fn(self.model, sequence)

    def train_sequence(self, index):
        """
        Construct the training sequence for the `index` of nodes.


        Parameters:
        ----------
        index: Numpy array-like, `list` or integer scalar
            The index of nodes used in training.

        Return:
        ----------
        sequence: The sequence of `graphgallery.Sequence` for the nodes.

        """

        raise NotImplementedError

    def test_sequence(self, index):
        """
        Construct the testing sequence for the `index` of nodes.

        Note:
        ----------
        If not implemented, this method will call `train_sequence` automatically.

        Parameters:
        ----------
        index: Numpy array-like, `list` or integer scalar
            The index of nodes used in testing.

        Return:
        ----------
        sequence: The sequence of `graphgallery.Sequence` for the nodes.
        """
        return self.train_sequence(index)

    def predict_sequence(self, index):
        """
        Construct the prediction sequence for the `index` of nodes.

        Note:
        ----------
            If not implemented, this method will call `test_sequence` automatically.

        Parameters:
        ----------
            index: Numpy array-like, `list` or integer scalar
                The index of nodes used in prediction.

        Return:
        ----------
            The sequence of `graphgallery.Sequence` for the nodes.
        """
        return self.test_sequence(index)

    def _test_predict(self, index):
        logit = self.predict(index)
        predict_class = logit.argmax(1)
        labels = self.graph.labels[index]
        return (predict_class == labels).mean()

    def reset_weights(self):
        # TODO: add torch support
        """reset the model to the first time.
        """
        model = self.model
        if self.backup is None:
            raise RuntimeError("You must store the `backup` before `reset_weights`."
                               "`backup` will be automatically stored when the model is built.")
        for w, wb in zip(model.weights, self.backup):
            w.assign(wb)

    def reset_optimizer(self):
        # TODO: add torch support
        model = self.model
        if hasattr(model, 'optimizer'):
            for var in model.optimizer.variables():
                var.assign(tf.zeros_like(var))

    def reset_lr(self, value):
        # TODO: add torch support
        model = self.model
        if not hasattr(model, 'optimizer'):
            raise RuntimeError("The model has not attribute `optimizer`!")
        model.optimizer.learning_rate.assign(value)


    def remove_weights(self):
        filepath = self.weight_path
        if not filepath.endswith(gg.file_postfix()):
            filepath = filepath + gg.file_postfix()

        if osp.exists(filepath):
            os.remove(filepath)
