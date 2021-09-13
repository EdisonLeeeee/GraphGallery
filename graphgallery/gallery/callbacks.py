# The following codes are mainly modified from tensorflow and some changes are made.
# You may refer to tensorflow for more details:
#
#     https://github.com/tensorflow/tensorflow
#
# Copyright The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import os
import logging
import numpy as np
from graphgallery import functional as gf


class Callback:
    """Abstract base class used to build new callbacks.

    Callbacks can be passed to keras methods such as `fit`, `evaluate`, and
    `predict` in order to hook into the various stages of the model training and
    inference lifecycle.

    To create a custom callback, subclass `keras.callbacks.Callback` and override
    the method associated with the stage of interest. See
    https://www.tensorflow.org/guide/keras/custom_callback for more information.

    Example:

    >>> training_finished = False
    >>> class MyCallback(tf.keras.callbacks.Callback):
    ...   def on_train_end(self, logs=None):
    ...     global training_finished
    ...     training_finished = True
    >>> model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
    >>> model.compile(loss='mean_squared_error')
    >>> model.fit(tf.constant([[1.0]]), tf.constant([[1.0]]),
    ...           callbacks=[MyCallback()])
    >>> assert training_finished == True

    Attributes:
        params: Dict. Training parameters
            (eg. verbosity, batch size, number of epochs...).
        model: Instance of `keras.models.Model`.
            Reference of the model being trained.

    The `logs` dictionary that callback methods
    take as argument will contain keys for quantities relevant to
    the current batch or epoch (see method-specific docstrings).
    """

    def __init__(self):
        self.validation_data = None  # pylint: disable=g-missing-from-attributes
        self.model = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_batch_begin(self, batch, logs=None):
        """A backwards compatibility alias for `on_train_batch_begin`."""

    def on_batch_end(self, batch, logs=None):
        """A backwards compatibility alias for `on_train_batch_end`."""

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch.

        Subclasses should override for any actions to run. This function should only
        be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict. Currently no data is passed to this argument for this method
              but that may change in the future.
        """

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch.

        Subclasses should override for any actions to run. This function should only
        be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict, metric results for this training epoch, and for the
              validation epoch if validation is performed. Validation result keys
              are prefixed with `val_`. For training epoch, the values of the
             `Model`'s metrics are returned. Example : `{'loss': 0.2, 'accuracy':
               0.7}`.
        """

    def on_train_batch_begin(self, batch, logs=None):
        """Called at the beginning of a training batch in `fit` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every `N`
        batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict, contains the return value of `model.train_step`. Typically,
              the values of the `Model`'s metrics are returned.  Example:
              `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        # For backwards compatibility.
        self.on_batch_begin(batch, logs=logs)

    def on_train_batch_end(self, batch, logs=None):
        """Called at the end of a training batch in `fit` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every `N`
        batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """
        # For backwards compatibility.
        self.on_batch_end(batch, logs=logs)

    def on_test_batch_begin(self, batch, logs=None):
        """Called at the beginning of a batch in `evaluate` methods.

        Also called at the beginning of a validation batch in the `fit`
        methods, if validation data is provided.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every `N`
        batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict, contains the return value of `model.test_step`. Typically,
              the values of the `Model`'s metrics are returned.  Example:
              `{'loss': 0.2, 'accuracy': 0.7}`.
        """

    def on_test_batch_end(self, batch, logs=None):
        """Called at the end of a batch in `evaluate` methods.

        Also called at the end of a validation batch in the `fit`
        methods, if validation data is provided.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every `N`
        batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """

    def on_predict_batch_begin(self, batch, logs=None):
        """Called at the beginning of a batch in `predict` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every `N`
        batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict, contains the return value of `model.predict_step`,
              it typically returns a dict with a key 'outputs' containing
              the model's outputs.
        """

    def on_predict_batch_end(self, batch, logs=None):
        """Called at the end of a batch in `predict` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every `N`
        batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """

    def on_train_begin(self, logs=None):
        """Called at the beginning of training.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument for this method
              but that may change in the future.
        """

    def on_train_end(self, logs=None):
        """Called at the end of training.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently the output of the last call to `on_epoch_end()`
              is passed to this argument for this method but that may change in
              the future.
        """

    def on_test_begin(self, logs=None):
        """Called at the beginning of evaluation or validation.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument for this method
              but that may change in the future.
        """

    def on_test_end(self, logs=None):
        """Called at the end of evaluation or validation.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently the output of the last call to
              `on_test_batch_end()` is passed to this argument for this method
              but that may change in the future.
        """

    def on_predict_begin(self, logs=None):
        """Called at the beginning of prediction.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument for this method
              but that may change in the future.
        """

    def on_predict_end(self, logs=None):
        """Called at the end of prediction.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument for this method
              but that may change in the future.
        """


class History(Callback):
    """Callback that records events into a `History` object.

    This callback is automatically applied to
    every Keras model. The `History` object
    gets returned by the `fit` method of models.

    Example:

    >>> model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
    >>> model.compile(tf.keras.optimizers.SGD(), loss='mse')
    >>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
    ...                     epochs=10)
    >>> print(history.params)
    {'verbose': 1, 'epochs': 10, 'steps': 1}
    >>> # check the keys of history object
    >>> print(history.history.keys())
    dict_keys(['loss'])

    """

    def __init__(self):
        super(History, self).__init__()
        self.history = gf.BunchDict()

    def on_train_begin(self, logs=None):
        self.epoch = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # Set the history attribute on the model after the epoch ends. This will
        # make sure that the state which is set is the latest one.
        self.model.history = self


class TerminateOnNaN(Callback):
    """Callback that terminates training when a NaN loss is encountered. """

    def __init__(self):
        super(TerminateOnNaN, self).__init__()
        self._supports_tf_logs = True

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None:
            loss = float(loss)
            if np.isnan(loss) or np.isinf(loss):
                print('Batch %d: Invalid loss, terminating training' % (batch))
                self.model.stop_training = True


class ModelCheckpoint(Callback):
    """Callback to save the Keras model or model weights at some frequency.

    `ModelCheckpoint` callback is used in conjunction with training using
    `model.fit()` to save a model or weights (in a checkpoint file) at some
    interval, so the model or weights can be loaded later to continue the training
    from the state saved.

    A few options this callback provides include:

    - Whether to only keep the model that has achieved the "best performance" so
      far, or whether to save the model at the end of every epoch regardless of
      performance.
    - Definition of 'best'; which quantity to monitor and whether it should be
      maximized or minimized.
    - The frequency it should save at. Currently, the callback supports saving at
      the end of every epoch, or after a fixed number of training batches.
    - Whether only weights are saved, or the whole model is saved.

    Note: If you get `WARNING:tensorflow:Can save best model only with <name>
    available, skipping` see the description of the `monitor` argument for
    details on how to get this right.

    Example:

    ```python
    model.compile(loss=..., optimizer=...,
                  metrics=['accuracy'])

    EPOCHS = 10
    checkpoint_filepath = '/tmp/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    # Model weights are saved at the end of every epoch, if it's the best seen
    # so far.
    model.fit(epochs=EPOCHS, callbacks=[model_checkpoint_callback])

    # The model weights (that are considered the best) are loaded into the model.
    model.load_weights(checkpoint_filepath)
    ```

    Args:
        filepath: string or `PathLike`, path to save the model file. e.g.
          filepath = os.path.join(working_dir, 'ckpt', file_name). `filepath`
          can contain named formatting options, which will be filled the value of
          `epoch` and keys in `logs` (passed in `on_epoch_end`). For example: if
          `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`, then the model
          checkpoints will be saved with the epoch number and the validation loss
          in the filename. The directory of the filepath should not be reused by
          any other callbacks to avoid conflicts.
        monitor: The metric name to monitor. Typically the metrics are set by the
          `Model.compile` method. Note:

          * Prefix the name with `"val_`" to monitor validation metrics.
          * Use `"loss"` or "`val_loss`" to monitor the model's total loss.
          * If you specify metrics as strings, like `"accuracy"`, pass the same
            string (with or without the `"val_"` prefix).
          * If you pass `metrics.Metric` objects, `monitor` should be set to
            `metric.name`
          * If you're not sure about the metric names you can check the contents
            of the `history.history` dictionary returned by
            `history = model.fit()`
          * Multi-output models set additional prefixes on the metric names.

        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`, it only saves when the model
          is considered the "best" and the latest best model according to the
          quantity monitored will not be overwritten. If `filepath` doesn't
          contain formatting options like `{epoch}` then `filepath` will be
          overwritten by each new better model.
        mode: one of {'auto', 'min', 'max'}. If `save_best_only=True`, the
          decision to overwrite the current save file is made based on either
          the maximization or the minimization of the monitored quantity.
          For `val_acc`, this should be `max`, for `val_loss` this should be
          `min`, etc. In `auto` mode, the mode is set to `max` if the quantities
          monitored are 'acc' or start with 'fmeasure' and are set to `min` for
          the rest of the quantities.
        save_weights_only: if True, then only the model's weights will be saved
          (`model.save_weights(filepath)`), else the full model is saved
          (`model.save(filepath)`).
        **kwargs: Additional arguments for backwards compatibility. 
    """

    def __init__(self,
                 filepath,
                 monitor='val_loss',
                 verbose=0,
                 save_best_only=False,
                 save_weights_only=False,
                 mode='auto',
                 **kwargs):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.epochs_since_last_save = 0
        self._batches_seen_since_last_saving = 0
        self._last_batch_seen = 0

        if mode not in ['auto', 'min', 'max']:
            logging.warning('ModelCheckpoint mode %s is unknown, '
                            'fallback to auto mode.', mode)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'loss' in self.monitor:
                self.monitor_op = np.less
                self.best = np.Inf
            else:
                self.monitor_op = np.greater
                self.best = -np.Inf

    def on_train_begin(self, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        self._save_model(epoch=epoch, logs=logs)

    def _save_model(self, epoch, logs):
        """Saves the model.

        Args:
            epoch: the epoch this iteration is in.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        logs = logs or {}

        filepath = self._get_file_path(epoch, logs)

        try:
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    logging.warning('Can save best model only with %s available, '
                                    'skipping.', self.monitor)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s' % (epoch + 1, self.monitor,
                                                           self.best, current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)
        except IOError as e:
            # `e.errno` appears to be `None` so checking the content of `e.args[0]`.
            if 'is a directory' in str(e.args[0]).lower():
                raise IOError('Please specify a non-directory filepath for '
                              'ModelCheckpoint. Filepath used is an existing '
                              'directory: {}'.format(filepath))
            # Re-throw the error for any other causes.
            raise e

    def _get_file_path(self, epoch, logs):
        """Returns the file path for checkpoint."""
        try:
            # `filepath` may contain placeholders such as `{epoch:02d}` and
            # `{mape:.2f}`. A mismatch between logged metrics and the path's
            # placeholders can cause formatting to fail.
            file_path = self.filepath.format(epoch=epoch + 1, **logs)
        except KeyError as e:
            raise KeyError('Failed to format this callback filepath: "{}". '
                           'Reason: {}'.format(self.filepath, e))
        return file_path


class LambdaCallback(Callback):
    r"""Callback for creating simple, custom callbacks on-the-fly.

    This callback is constructed with anonymous functions that will be called
    at the appropriate time (during `Model.{fit | evaluate | predict}`).
    Note that the callbacks expects positional arguments, as:

    - `on_epoch_begin` and `on_epoch_end` expect two positional arguments:
      `epoch`, `logs`
    - `on_batch_begin` and `on_batch_end` expect two positional arguments:
      `batch`, `logs`
    - `on_train_begin` and `on_train_end` expect one positional argument:
      `logs`

    Args:
        on_epoch_begin: called at the beginning of every epoch.
        on_epoch_end: called at the end of every epoch.
        on_batch_begin: called at the beginning of every batch.
        on_batch_end: called at the end of every batch.
        on_train_begin: called at the beginning of model training.
        on_train_end: called at the end of model training.

    Example:

    ```python
    # Print the batch number at the beginning of every batch.
    batch_print_callback = LambdaCallback(
        on_batch_begin=lambda batch,logs: print(batch))

    # Stream the epoch loss to a file in JSON format. The file content
    # is not well-formed JSON but rather has a JSON object per line.
    import json
    json_log = open('loss_log.json', mode='wt', buffering=1)
    json_logging_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: json_log.write(
            json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '\n'),
        on_train_end=lambda logs: json_log.close()
    )

    # Terminate some processes after having finished model training.
    processes = ...
    cleanup_callback = LambdaCallback(
        on_train_end=lambda logs: [
            p.terminate() for p in processes if p.is_alive()])

    model.fit(...,
              callbacks=[batch_print_callback,
                         json_logging_callback,
                         cleanup_callback])
    ```
    """

    def __init__(self,
                 on_epoch_begin=None,
                 on_epoch_end=None,
                 on_batch_begin=None,
                 on_batch_end=None,
                 on_train_begin=None,
                 on_train_end=None,
                 **kwargs):
        super(LambdaCallback, self).__init__()
        self.__dict__.update(kwargs)
        if on_epoch_begin is not None:
            self.on_epoch_begin = on_epoch_begin
        else:
            self.on_epoch_begin = lambda epoch, logs: None
        if on_epoch_end is not None:
            self.on_epoch_end = on_epoch_end
        else:
            self.on_epoch_end = lambda epoch, logs: None
        if on_batch_begin is not None:
            self.on_batch_begin = on_batch_begin
        else:
            self.on_batch_begin = lambda batch, logs: None
        if on_batch_end is not None:
            self.on_batch_end = on_batch_end
        else:
            self.on_batch_end = lambda batch, logs: None
        if on_train_begin is not None:
            self.on_train_begin = on_train_begin
        else:
            self.on_train_begin = lambda logs: None
        if on_train_end is not None:
            self.on_train_end = on_train_end
        else:
            self.on_train_end = lambda logs: None
