import os
import sys
import time
import copy
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
from graphgallery.functional import Bunch
from graphgallery.nn.functions import softmax
from graphgallery.data.io import makedirs_from_filepath
from graphgallery.data import BaseGraph
from graphgallery.utils.raise_error import raise_if_kwargs
from graphgallery.utils import trainer
from graphgallery.gallery import GraphModel

# Ignora warnings:
#     UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
#     This is caused by `tf.gather` and it will be solved in future tensorflow version.
warnings.filterwarnings(
    'ignore',
    '.*Converting sparse IndexedSlices to a dense Tensor of unknown shape.*')


class GalleryModel(GraphModel):
    def __init__(self, *graph, device='cpu:0', seed=None, name=None, **kwargs):
        super().__init__(*graph, device=device, seed=seed, name=name, **kwargs)

        if self.backend == "tensorflow":
            self.train_step_fn = partial(trainer.train_step_tf,
                                         device=self.device)
            self.test_step_fn = partial(trainer.test_step_tf,
                                        device=self.device)
            self.predict_step_fn = partial(trainer.predict_step_tf,
                                           device=self.device)
        else:
            self.train_step_fn = trainer.train_step_torch
            self.test_step_fn = trainer.test_step_torch
            self.predict_step_fn = trainer.predict_step_torch

    def process(self, graph=None, updates=None, **kwargs):
        """pre-process for the input graph, including manipulations
        on adjacency matrix and node node attribute matrix, and finally convert
        them into tensor (optional).

        Note:
        ----------
        This method will call the method 'process_step'
            and it must be implemented for processing the graph.

        Parameters:
        ----------
        graph: An instance of graphgallery graph.
        updates: dict, the updates for the graph
        kwargs: other custom keyword parameters for 'process_step'.
        """
        assert not graph or isinstance(graph, BaseGraph)
        if graph is not None:
            assert isinstance(graph, BaseGraph)
            self.graph = graph
        if updates is not None:
            assert isinstance(updates, dict)
            self.graph.update(**updates)

        return self.process_step(**kwargs)

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
                The number of hidden units of model. Note: the last hidden unit (`num_node_classes`)
                aren't necessary to specified and it will be automatically added in the last
                layer.
            activations: `list` of string or string
                The activation function of model. Note: the last activation function (`softmax`)
                aren't necessary to specified and it will be automatically specified in the
                final output.
            dropout: float scalar
                Dropout rate for the hidden outputs.
            weight_decay:  float scalar
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

    def train(self,
              train_data,
              val_data=None,
              epochs=200,
              early_stopping=None,
              verbose=1,
              save_best=True,
              ckpt_path=None,
              as_model=False,
              monitor='val_accuracy',
              early_stop_metric='val_loss',
              callbacks=None,
              **kwargs):
        """Train the model for the input `train_data` of nodes or `sequence`.

        Note:
        ----------
        You must compile your model before training/testing/predicting. Use `model.build()`.

        Parameters:
        ----------
        train_data: Numpy array-like, `list`, Integer scalar or `graphgallery.Sequence`
            The index of nodes (or sequence) that will be used during training.
        val_data: Numpy array-like, `list`, Integer scalar or
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
            (default :obj: 1)
        save_best: bool
            Whether to save the best weights (accuracy of loss depend on `monitor`)
            of training or validation (depend on `validation` is `False` or `True`).
            (default :bool: `True`)
        ckpt_path: String or None
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
        if not (isinstance(verbose, int) and 0 <= verbose <= 4):
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
                'You must compile your model before training/testing/predicting. Use `model.build()`.'
            )

        metrics_names = getattr(model, "metrics_names", None)
        # FIXME: This would be return '[]' for tensorflow>=2.2.0
        # See <https://github.com/tensorflow/tensorflow/issues/37990>
        if not metrics_names:
            raise RuntimeError(f"Please specify the attribute 'metrics_names' for the model.")
        if not isinstance(train_data, Sequence):
            train_data = self.train_sequence(train_data)

        self.train_data = train_data

        validation = val_data is not None

        if validation:
            if not isinstance(val_data, Sequence):
                val_data = self.test_sequence(val_data)
            self.val_data = val_data
            metrics_names = metrics_names + ["val_" + metric for metric in metrics_names]

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
            if not ckpt_path:
                ckpt_path = self.ckpt_path
            else:
                self.ckpt_path = ckpt_path

            makedirs_from_filepath(ckpt_path)

            if not ckpt_path.endswith(gg.file_ext()):
                ckpt_path = ckpt_path + gg.file_ext()

            assert monitor in metrics_names, f"'{monitor}' are not included in the metrics names."
            mc_callback = ModelCheckpoint(ckpt_path,
                                          monitor=monitor,
                                          save_best_only=True,
                                          save_weights_only=not as_model,
                                          verbose=0)
            callbacks.append(mc_callback)

        callbacks.set_model(model)
        model.stop_training = False

        metrics_names = metrics_names + ["Duration"]
        if verbose:
            stateful_metrics = set(metrics_names)
            if verbose <= 2:
                progbar = Progbar(target=epochs,
                                  verbose=verbose,
                                  stateful_metrics=stateful_metrics)
            print("Training...")

        logs = Bunch()
        begin_time = time.perf_counter()
        callbacks.on_train_begin()
        try:
            for epoch in range(epochs):
                if verbose > 2:
                    progbar = Progbar(target=len(train_data),
                                      verbose=verbose - 2,
                                      stateful_metrics=stateful_metrics)

                callbacks.on_epoch_begin(epoch)
                callbacks.on_train_batch_begin(0)
                train_logs = self.train_step(train_data)
                train_data.on_epoch_end()

                logs.update(train_logs)

                if validation:
                    valid_logs = self.test_step(val_data)
                    logs.update({("val_" + k): v for k, v in valid_logs.items()})
                    val_data.on_epoch_end()

                callbacks.on_train_batch_end(len(train_data), logs)
                callbacks.on_epoch_end(epoch, logs)

                time_passed = time.perf_counter() - begin_time
                logs["Duration"] = time_passed

                if verbose > 2:
                    print(f"Epoch {epoch+1}/{epochs}")
                    progbar.update(len(train_data), logs.items())
                elif verbose:
                    progbar.update(epoch + 1, logs.items())

                if model.stop_training:
                    print(f"Early Stopping in Epoch {epoch}", file=sys.stderr)
                    break

            callbacks.on_train_end()
            self.load(ckpt_path, as_model=as_model)
        finally:
            # to avoid unexpected termination of the model
            self.remove_weights()

        return history

    def test(self, data, verbose=1):
        """Test the output accuracy for the data.

        Note:
        ----------
        You must compile your model before training/testing/predicting.
        Use `model.build()`.

        Parameters:
        ----------
        data: Numpy array-like, `list` or `graphgallery.Sequence`
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
                'You must compile your model before training/testing/predicting. Use `model.build()`.'
            )

        if isinstance(data, Sequence):
            test_data = data
        else:
            test_data = self.test_sequence(data)

        self.test_data = test_data

        if verbose:
            print("Testing...")

        metrics_names = self.model.metrics_names + ["Duration"]

        progbar = Progbar(target=len(test_data),
                          verbose=verbose,
                          stateful_metrics=set(metrics_names))
        begin_time = time.perf_counter()
        logs = Bunch(**self.test_step(test_data))
        time_passed = time.perf_counter() - begin_time
        logs["Duration"] = time_passed
        progbar.update(len(test_data), logs.items())
        return logs

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

    def predict(self, predict_data=None, return_prob=True):
        """
        Predict the output probability for the input data.

        Note:
        ----------
        You must compile your model before training/testing/predicting.
            Use `model.build()`.

        Parameters:
        ----------
        predict_data: Numpy 1D array, optional.
            The indices of nodes to predict.
            if None, predict the all nodes.

        return_prob: bool.
            whether to return the probability of prediction.

        Return:
        ----------
        The predicted probability of each class for each node,
            shape (num_nodes, num_node_classes).

        """

        if not self.model:
            raise RuntimeError(
                'You must compile your model before training/testing/predicting. Use `model.build()`.'
            )

        if predict_data is None:
            predict_data = np.arange(self.graph.num_nodes, dtype=gg.intx())

        if not isinstance(predict_data, Sequence):
            predict_data = self.predict_sequence(predict_data)

        self.predict_data = predict_data

        logit = self.predict_step(predict_data)
        if return_prob:
            logit = softmax(logit)
        return logit

    def predict_step(self, sequence):
        return self.predict_step_fn(self.model, sequence)

    def train_sequence(self, inputs):
        """
        Construct the training sequence.
        """

        raise NotImplementedError

    def test_sequence(self, inputs):
        """
        Construct the testing sequence.

        Note:
        ----------
        If not implemented, this method will call `train_sequence` automatically.
        """
        return self.train_sequence(inputs)

    def predict_sequence(self, inputs):
        """
        Construct the prediction sequence.
        Note:
        ----------
        If not implemented, this method will call `train_sequence` automatically.        
        """
        return self.test_sequence(inputs)

    def _test_predict(self, index):
        logit = self.predict(index)
        predict_class = logit.argmax(1)
        labels = self.graph.node_label[index]
        return (predict_class == labels).mean()

    def reset_weights(self):
        # TODO: add pytorch support
        """reset the model to the first time.
        """
        model = self.model
        if self.backup is None:
            raise RuntimeError(
                "You must store the `backup` before `reset_weights`."
                "`backup` will be automatically stored when the model is built."
            )
        for w, wb in zip(model.weights, self.backup):
            w.assign(wb)

    def reset_optimizer(self):
        # TODO: add pytorch support
        model = self.model
        if hasattr(model, 'optimizer'):
            for var in model.optimizer.variables():
                var.assign(tf.zeros_like(var))

    def reset_lr(self, value):
        # TODO: add pytorch support
        model = self.model
        if not hasattr(model, 'optimizer'):
            raise RuntimeError("The model has not attribute `optimizer`!")
        model.optimizer.learning_rate.assign(value)

    def remove_weights(self):
        filepath = self.ckpt_path
        if self.backend == "tensorflow":
            remove_extra_tf_files(filepath)

        ext = gg.file_ext()
        if not filepath.endswith(ext):
            filepath = filepath + ext

        if osp.exists(filepath):
            os.remove(filepath)


def remove_extra_tf_files(filepath):
    # for tensorflow weights that saved without h5 formate
    for ext in (".data-00000-of-00001", ".data-00000-of-00002",
                ".data-00001-of-00002", ".index"):
        path = filepath + ext
        if osp.exists(path):
            os.remove(path)

    file_dir = osp.split(osp.realpath(filepath))[0]

    path = osp.join(file_dir, "checkpoint")
    if osp.exists(path):
        os.remove(path)
