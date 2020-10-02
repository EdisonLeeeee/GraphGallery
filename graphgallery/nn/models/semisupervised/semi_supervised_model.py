import os
import time
import copy
import logging
import warnings
import torch
import os.path as osp
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from functools import partial

from tensorflow.keras.utils import Sequence
from tensorflow.python.keras import callbacks as callbacks_module
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ProgbarLogger
from tensorflow.keras.callbacks import History as tf_History
from tensorflow.python.keras import callbacks as cbks

from graphgallery.nn.models import BaseModel
from graphgallery.nn.functions import softmax
from graphgallery.utils.history import History
from graphgallery.utils.tqdm import tqdm
from graphgallery.data.io import makedirs_from_path
from graphgallery.utils.raise_error import raise_if_kwargs
from graphgallery.data import Basegraph
from graphgallery.transforms import asintarr

# Ignora warnings:
#     UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
#     This is caused by `tf.gather` and it will be solved in future tensorflow version.
warnings.filterwarnings(
    'ignore', '.*Converting sparse IndexedSlices to a dense Tensor of unknown shape.*')


class SemiSupervisedModel(BaseModel):
    def __init__(self, *graph, device='cpu:0', seed=None, name=None, **kwargs):
        super().__init__(*graph, device=device, seed=seed, name=name, **kwargs)
    
        if self.kind == "T":
            self.train_step_fn = partial(train_step_tf, device=self.device)
            self.test_step_fn = partial(test_step_tf, device=self.device)
            self.predict_step_fn = partial(predict_step_tf, device=self.device)
        else:
            self.train_step_fn = train_step_torch
            self.test_step_fn =test_step_torch
            self.predict_step_fn =predict_step_torch
        
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
        kwargs: other customized keyword Parameters.

        """
        if len(graph) > 0:
            if len(graph) == 1:
                graph, = graph
                if isinstance(graph, Basegraph):
                    self.graph = graph
                elif isinstance(graph, dict):
                    self.graph.set_inputs(**graph)
            else:
                self.graph.set_inputs(*graph)
                
        return self.process_step()

    def process_step(self):
        raise NotImplementedError

    def build(self):
        """Build the model using customized hyperparameters.

        Note:
        ----------
            This method must be called before training/testing/predicting.
            Use `model.build()`. The following `Parameters` are only commonly used
            Parameters, and other model-specific Parameters are not introduced as follows.

        Parameters:
        ----------
            hiddens: `list` of integer or integer scalar
                The number of hidden units of model. Note: the last hidden unit (`n_classes`)
                aren't nececcary to specified and it will be automatically added in the last
                layer.
            activations: `list` of string or string
                The activation function of model. Note: the last activation function (`softmax`)
                aren't nececcary to specified and it will be automatically spefified in the
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
        """Build the model using customized model.

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
        if self.kind == "T":
            self.model = model.to(self.device)
        else:
            with tf.device(self.device):
                self.model = model

    def train_v1(self, idx_train, idx_val=None,
                 epochs=200, early_stopping=None,
                 verbose=False, save_best=True, weight_path=None, as_model=False,
                 monitor='val_acc', early_stop_metric='val_loss'):
        """Train the model for the input `idx_train` of nodes or `sequence`.

        Note:
        ----------
            You must compile your model before training/testing/predicting. Use `model.build()`.

        Parameters:
        ----------
        idx_train: Numpy array-like, `list`, Integer scalar or
            `graphgallery.Sequence`.
            The index of nodes (or sequence) that will be used during training.
        idx_val: Numpy array-like, `list`, Integer scalar or
            `graphgallery.Sequence`, optional
            The index of nodes (or sequence) that will be used for validation.
            (default :obj: `None`, i.e., do not use validation during training)
        epochs: integer
            The number of epochs of training.(default :obj: `200`)
        early_stopping: integer or None
            The number of early stopping patience during training. (default :obj: `None`,
            i.e., do not use early stopping during training)
        verbose: bool
            Whether to show the training details. (default :obj: `None`)
        save_best: bool
            Whether to save the best weights (accuracy of loss depend on `monitor`)
            of training or validation (depend on `validation` is `False` or `True`).
            (default :bool: `True`)
        weight_path: String or None
            The path of saved weights/model. (default :obj: `None`, i.e.,
            `./log/{self.name}_weights`)
        as_model: bool
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

        # Check if model has been built
        if self.model is None:
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

        history = History(monitor_metric=monitor,
                          early_stop_metric=early_stop_metric)

        if not weight_path:
            weight_path = self.weight_path

        if validation is None:
            history.register_monitor_metric('acc')
            history.register_early_stop_metric('loss')

        if verbose:
            pbar = tqdm(range(1, epochs + 1))
        else:
            pbar = range(1, epochs + 1)

        for epoch in pbar:

            loss, accuracy = self.train_step(train_data)
            train_data.on_epoch_end()

            history.add_results(loss, 'loss')
            history.add_results(accuracy, 'acc')

            if validation:

                val_loss, val_accuracy = self.test_step(val_data)
                val_data.on_epoch_end()
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
            if self.kind == "T":
                remove_tf_weights(weight_path)
            else:
                remove_torch_weights(weight_path)

        return history

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
        verbose: int in {0, 1, 2}
                'verbose=0': not verbose; 
                'verbose=1': tqdm verbose; 
                'verbose=2': tensorflow probar verbose;        
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
            must be speficied if you are using customized `layer` or `loss` and so on.
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
        if not verbose in {0, 1, 2}:
            raise ValueError("'verbose=0': not verbose; 'verbose=1': tqdm verbose; "
                             "'verbose=2': tensorflow probar verbose; "
                             f"but got {verbose}")
        model = self.model
        # Check if model has been built
        if model is None:
            raise RuntimeError(
                'You must compile your model before training/testing/predicting. Use `model.build()`.')

        # TODO: add metric names in `model`
        metric_names = ['loss', 'acc']
        callback_metrics = metric_names
        model.stop_training = False

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
            callback_metrics = copy.copy(metric_names)
            callback_metrics += ['val_' + n for n in metric_names]
        else:
            monitor = 'acc' if monitor[:3] == 'val' else monitor

        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(callbacks)

        history = tf_History()
        callbacks.append(history)
        
        if verbose == 2:
            callbacks.append(ProgbarLogger(stateful_metrics=metric_names[1:]))


        if early_stopping:
            es_callback = EarlyStopping(monitor=early_stop_metric,
                                        patience=early_stopping,
                                        mode='auto',
                                        verbose=kwargs.pop('es_verbose', 1))
            callbacks.append(es_callback)

        if save_best:
            if not weight_path:
                weight_path = self.weight_path

            makedirs_from_path(weight_path)

            if not weight_path.endswith('.h5'):
                weight_path = weight_path + '.h5'

            mc_callback = ModelCheckpoint(weight_path,
                                          monitor=monitor,
                                          save_best_only=True,
                                          save_weights_only=not as_model,
                                          verbose=0)
            callbacks.append(mc_callback)
            
        callbacks.set_model(model)
        # TODO: to be improved
        callback_params = {
            'batch_size': None,
            'epochs': epochs,
            'steps': 1,
            'samples': 1,
            'verbose': verbose==2,
            'do_validation': validation,
            'metrics': callback_metrics,
        }
        callbacks.set_params(callback_params)
        raise_if_kwargs(kwargs)

        callbacks.on_train_begin()

        if verbose == 1:
            pbar = tqdm(range(1, epochs + 1))
        else:
            pbar = range(epochs)

        for epoch in pbar:
            callbacks.on_epoch_begin(epoch)

            callbacks.on_train_batch_begin(0)
            loss, accuracy = self.train_step(train_data)

            training_logs = {'loss': loss, 'acc': accuracy}

            if validation:
                val_loss, val_accuracy = self.test_step(val_data)
                training_logs.update(
                    {'val_loss': val_loss, 'val_acc': val_accuracy})
                val_data.on_epoch_end()
            callbacks.on_train_batch_end(0, training_logs)
            callbacks.on_epoch_end(epoch, training_logs)

            if verbose == 1:
                msg = "<"
                for key, val in training_logs.items():
                    msg += f"{key.title()} = {val:.4f} "
                msg += ">"
                pbar.set_description(msg)
            train_data.on_epoch_end()
            
            if verbose == 2:
                print()
                
            if model.stop_training:
                break

        callbacks.on_train_end()

        if save_best:
            self.load(weight_path, as_model=as_model)
            remove_tf_weights(weight_path)

        return history

    def train_v2(self, idx_train, idx_val=None,
                 epochs=200, early_stopping=None,
                 verbose=False, save_best=True, weight_path=None, as_model=False,
                 monitor='val_acc', early_stop_metric='val_loss', callbacks=None, **kwargs):
        """
            Train the model for the input `idx_train` of nodes or `sequence`.

        Note:
        ----------
        You must compile your model before training/testing/predicting. Use `model.build()`.

        Parameters:
        ----------
        idx_train: Numpy array-like, `list`, Integer scalar or
            `graphgallery.Sequence`.
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
        verbose: bool
            Whether to show the training details. (default :obj: `None`)
        save_best: bool
            Whether to save the best weights (accuracy of loss depend on `monitor`)
            of training or validation (depend on `validation` is `False` or `True`).
            (default :bool: `True`)
        weight_path: String or None
            The path of saved weights/model. (default :obj: `None`, i.e.,
            `./log/{self.name}_weights`)
        as_model: bool
            Whether to save the whole model or weights only, if `True`, the `self.custom_objects`
            must be speficied if you are using customized `layer` or `loss` and so on.
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

        if not tf.__version__ >= '2.2.0':
            raise RuntimeError(
                f'This method is only work for tensorflow version >= 2.2.0.')

        # Check if model has been built
        if self.model is None:
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
                weight_path = self.weight_path

            makedirs_from_path(weight_path)

            if not weight_path.endswith('.h5'):
                weight_path += '.h5'

            mc_callback = ModelCheckpoint(weight_path,
                                          monitor=monitor,
                                          save_best_only=True,
                                          save_weights_only=not as_model,
                                          verbose=0)
            callbacks.append(mc_callback)
        callbacks.set_model(model)

        # leave it blank for the future
        allowed_kwargs = set([])
        unknown_kwargs = set(kwargs.keys()) - allowed_kwargs
        if unknown_kwargs:
            raise TypeError(
                "Invalid keyword argument(s): %s" % (unknown_kwargs,))

        callbacks.on_train_begin()

        for epoch in range(epochs):
            callbacks.on_epoch_begin(epoch)

            callbacks.on_train_batch_begin(0)
            loss, accuracy = self.train_step(train_data)
            train_data.on_epoch_end()

            training_logs = {'loss': loss, 'acc': accuracy}
            callbacks.on_train_batch_end(0, training_logs)

            if validation:

                val_loss, val_accuracy = self.test_step(val_data)
                training_logs.update(
                    {'val_loss': val_loss, 'val_acc': val_accuracy})
                val_data.on_epoch_end()

            callbacks.on_epoch_end(epoch, training_logs)

            if model.stop_training:
                break

        callbacks.on_train_end()

        if save_best:
            self.load(weight_path, as_model=as_model)
            remove_tf_weights(weight_path)

        return model.history

    def test(self, index):
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

        # TODO record test logs like self.train()
        if not self.model:
            raise RuntimeError(
                'You must compile your model before training/testing/predicting. Use `model.build()`.')

        if isinstance(index, Sequence):
            test_data = index
        else:
            index = asintarr(index)
            test_data = self.test_sequence(index)
            self.idx_test = index

        loss, accuracy = self.test_step(test_data)

        return loss, accuracy

    def train_step(self, sequence):
        """
        Forward propagation for the input `sequence`. This method will be called
        in `train`. If you want to specify your customized data during training/testing/predicting,
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
        in `test`. If you want to specify your customized data during training/testing/predicting,
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
            index = np.arange(self.graph.n_nodes)
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


def train_step_tf(model, sequence, device):
    model.reset_metrics()

    with tf.device(device):
        for inputs, labels in sequence:
            loss, accuracy = model.train_on_batch(
                x=inputs, y=labels, reset_metrics=False)

    return loss, accuracy

# def train_step_tf(model, sequence, device):
#     model.reset_metrics()
#     loss_fn = model.loss
#     metric = model.metrics[0]
#     optimizer = model.optimizer
#     model.reset_metrics()
#     metric.reset_states()

#     loss = 0.
#     with tf.GradientTape() as tape:
#         for inputs, labels in sequence:
#             output = model(inputs, training=True)
#             loss += loss_fn(labels, output)
#             metric.update_state(labels, output)

#     grad = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(grad, model.trainable_variables))

#     return loss, metric.result()

def train_step_torch(model, sequence):
    model.train()
    optimizer = model.optimizer
    loss_fn = model.loss_fn

    accuracy = 0.
    loss = 0.
    n_inputs = 0

    for inputs, labels in sequence:
        optimizer.zero_grad()
        output = model(inputs)
        _loss = loss_fn(output, labels)
        _loss.backward()
        optimizer.step()
        with torch.no_grad():
            loss += _loss.data
            accuracy += (output.argmax(1) == labels).float().sum()
            n_inputs += labels.size(0)

    return loss.detach().item(), (accuracy / n_inputs).detach().item()


def test_step_tf(model, sequence, device):
    model.reset_metrics()

    with tf.device(device):
        for inputs, labels in sequence:
            loss, accuracy = model.test_on_batch(
                x=inputs, y=labels, reset_metrics=False)

    return loss, accuracy

# def test_step_tf(model, sequence, device):
#     model.reset_metrics()
#     loss_fn = model.loss
#     metric = model.metrics[0]
#     optimizer = model.optimizer
#     model.reset_metrics()

#     loss = 0.
#     for inputs, labels in sequence:
#         output = model(inputs, training=False)
#         loss += loss_fn(labels, output)
#         metric.update_state(labels, output)

#     return loss, metric.result()


@torch.no_grad()
def test_step_torch(model, sequence):
    model.eval()
    loss_fn = model.loss_fn
    accuracy = 0.
    loss = 0.
    n_inputs = 0

    for inputs, labels in sequence:
        output = model(inputs)
        _loss = loss_fn(output, labels)
        loss += _loss.data 
        n_inputs += labels.size(0)
        accuracy += (output.argmax(1) == labels).float().sum()

    return loss.detach().item(), (accuracy / n_inputs).detach().item()


def predict_step_tf(model, sequence, device):
    logits = []
    with tf.device(device):
        for inputs, *_ in sequence:
            logit = model.predict_on_batch(x=inputs)
            if tf.is_tensor(logit):
                logit = logit.numpy()
            logits.append(logit)

    if len(sequence) > 1:
        logits = np.vstack(logits)
    else:
        logits = logits[0]
    return logits


# def predict_step_tf(model, sequence, device):
#     logits = []
#     with tf.device(device):
#         for inputs, *_ in sequence:
#             logit = model(inputs, training=False)
#             logits.append(logit)

#     if len(sequence) > 1:
#         logits = tf.concat(logits, axis=0)
#     else:
#         logits = logits[0]

#     return logits.numpy()

@torch.no_grad()
def predict_step_torch(model, sequence):
    model.eval()
    logits = []

    for inputs, _ in sequence:
        logit = model(inputs)
        logits.append(logit)

    if len(sequence) > 1:
        logits = torch.cat(logits)
    else:
        logits, = logits

    return logits.detach().cpu().numpy()


_POSTFIX = (".h5", ".data-00000-of-00001", ".index")


def remove_tf_weights(filepath_without_h5):
    if filepath_without_h5.endswith('.h5'):
        filepath_without_h5 = filepath_without_h5[:-3]

    # for tensorflow weights that saved without h5 formate
    for postfix in _POSTFIX:
        path = filepath_without_h5 + postfix
        if osp.exists(path):
            os.remove(path)

    file_dir = osp.split(osp.realpath(filepath_without_h5))[0]

    path = osp.join(file_dir, "checkpoint")
    if osp.exists(path):
        os.remove(path)


def remove_torch_weights(filepath):
    if not filepath.endswith('.pt'):
        filepath_with_pt = filepath + '.pt'

    if osp.exists(filepath_with_pt):
        os.remove(filepath_with_pt)
