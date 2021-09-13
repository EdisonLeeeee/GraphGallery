import os
import sys
import warnings
import importlib
import inspect
import os.path as osp
import numpy as np

from graphgallery.gallery.callbacks import ModelCheckpoint, CallbackList
from torch.utils.data import DataLoader, Dataset

import graphgallery as gg
from graphgallery import functional as gf
from graphgallery.data.io import makedirs_from_filepath
from graphgallery.gallery import Model
from graphgallery.utils import Progbar


# TensorFlow 2.1.x
# Ignora warnings:
#     UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
#     This is caused by `tf.gather` and it will be solved in future tensorflow version.
warnings.filterwarnings(
    'ignore',
    message='.*Converting sparse IndexedSlices to a dense Tensor of unknown shape.*')

# TensorFlow 2.4.0
# Ignora warnings:
#     UserWarning: Converting sparse IndexedSlices(IndexedSlices(indices=...) to a dense Tensor of unknown shape.
#     This may consume a large amount of memory.
warnings.filterwarnings(
    'ignore', message='.*to a dense Tensor of unknown shape.*')


def format_doc(d):
    msg = ""
    for i, (k, v) in enumerate(d.items()):
        if v != "UNSPECIDIED":
            msg += f"({i + 1}) `{k}`, Default is `{v}` \n"
        else:
            msg += f"({i + 1}) `{k}`, UNSPECIDIED argument\n"
    return msg


def doc_dict(func):
    ArgSpec = inspect.getfullargspec(func)
    args = ArgSpec.args if ArgSpec.args else []
    args = args[1:] if args[0] == "self" else args
    defaults = ArgSpec.defaults if ArgSpec.defaults else []
    delta_l = len(args) - len(defaults)
    defaults = ["UNSPECIDIED"] * delta_l + list(defaults)
    d = dict(zip(args, defaults))
    return d


def make_docs(*func):
    d = {}
    for f in func:
        d.update(doc_dict(f))
    return format_doc(d)


def unravel_batch(batch):
    inputs = labels = out_index = None
    if isinstance(batch, (list, tuple)):
        inputs = batch[0]
        labels = batch[1]
        if len(batch) > 2:
            out_index = batch[-1]
    else:
        inputs = batch

    if isinstance(labels, (list, tuple)) and len(labels) == 1:
        labels = labels[0]
    if isinstance(out_index, (list, tuple)) and len(out_index) == 1:
        out_index = out_index[0]

    return inputs, labels, out_index


class Trainer(Model):
    def setup_cfg(self):
        """load the default config function `default_cfg_setup` for the corresponding task.

        Raises
        ------
        RuntimeError
            the default config function `default_cfg_setup` not found in the file `graphgallery.gallery.[task].default`
        """
        # nodeclas/linkpred/...
        task_module = self.__module__.split('.')[2]
        # graphgallery.gallery
        gallery_module = '.'.join(__name__.split('.')[:-1])
        try:
            default_setup = importlib.import_module(f".{task_module}.default", gallery_module)
        except ModuleNotFoundError:
            raise RuntimeError(f"default setup function `{gallery_module}.{task_module}.default.default_cfg_setup` not found!")

        default_setup.default_cfg_setup(self.cfg)

    @np.deprecate(old_name="make_data",
                  message=("the method `trainer.make_data` is currently deprecated from 0.9.0,"
                           " please use `trainer.setup_graph` instead."))
    def make_data(self, *args, **kwargs):
        return self.setup_graph(*args, **kwargs)

    def setup_graph(self, graph, graph_transform=None, device=None, **kwargs):
        """This method is used for process your inputs, which accepts
        only keyword arguments in your defined method 'data_step'.
        This method will process the inputs, and transform them into tensors.

        Commonly used keyword arguments:
        --------------------------------
        graph: graphgallery graph classes.
        graph_transform: string, Callable function,
            or a tuple with function and dict arguments.
            transform for the entire graph, it is used first.
        device: device for preparing data, if None, it defaults to `self.device`
        adj_transform: string, Callable function,
            or a tuple with function and dict arguments.
            transform for adjacency matrix.
        attr_transform: string, Callable function,
            or a tuple with function and dict arguments.
            transform for attribute matrix.
        other arguments (if have) will be passed into method 'data_step'.
        """
        self.empty_cache()
        model = self.model
        if model is not None and hasattr(model, 'empty_cache'):
            model.empty_cache()

        self.graph = gf.get(graph_transform)(graph)
        cfg = self.cfg.data
        if device is not None:
            self.data_device = gf.device(device, self.backend)
        else:
            self.data_device = self.device
        cfg.device = device
        _, kwargs = gf.wrapper(self.data_step)(**kwargs)
        kwargs['graph_transform'] = graph_transform
        cfg.merge_from_dict(kwargs)

        for k, v in kwargs.items():
            if k.endswith("transform"):
                setattr(self.transform, k, gf.get(v))

        return self

    def data_step(self, *args, **kwargs):
        """Implement you data processing function here"""
        raise NotImplementedError

    def build(self, **kwargs):
        """This method is used for build your model, which
        accepts only keyword arguments in your defined method 'model_step'.

        Note:
        -----
        This method should be called after `process`.

        Commonly used keyword arguments:
        --------------------------------
        hids: int or a list of them,
            hidden units for each hidden layer.
        acts: string or a list of them,
            activation functions for each layer.
        dropout: float scalar,
            dropout used in the model.
        lr: float scalar,
            learning rate used for the model.
        weight_decay: float scalar,
            weight decay used for the model weights.
        bias: bool,
            whether to use bias in each layer.
        use_tfn: bool,
            this argument is only used for TensorFlow backend, if `True`, it will decorate
            the model training and testing with `tf.function` (See `graphgallery.nn.modes.TFKeras`).
            By default, it was `True`, which can accelerate the training and inference, by it may cause
            several errors.
        other arguments (if have) will be passed into your method 'model_step'.
        """
        if self._graph is None:
            raise RuntimeError("Please call 'trainer.setup_graph(graph)' first.")

        use_tfn = kwargs.get("use_tfn", True)
        if self.backend == "tensorflow":
            with self.backend.device(self.device):
                self.model, kwargs = gf.wrapper(self.model_step)(**kwargs)
                if use_tfn:
                    self.model.use_tfn()
        else:
            kwargs.pop("use_tfn", None)
            model, kwargs = gf.wrapper(self.model_step)(**kwargs)
            self.model = model.to(self.device)
        self.cfg.model.merge_from_dict(kwargs)
        return self

    def model_step(self, *args, **kwargs):
        """Implement you model building function here"""
        raise NotImplementedError

    def fit(self, train_data, val_data=None, **kwargs):

        cache = self.cache
        cfg = self.cfg.fit
        cfg.merge_from_dict(kwargs)
        ckpt_cfg = cfg.ModelCheckpoint
        model = self.model

        if model is None:
            raise RuntimeError(
                'You must compile your model before training/testing/predicting. Use `trainer.build()`.'
            )

        if not isinstance(train_data, (DataLoader, Dataset)):
            train_data = self.train_loader(train_data)

        if cfg.cache_train_data:
            cache.train_data = train_data

        validation = val_data is not None
        if validation:
            if not isinstance(val_data, (DataLoader, Dataset)):
                val_data = self.test_loader(val_data)
            if cfg.cache_val_data:
                cache.val_data = val_data

        # Setup callbacks
        verbose = cfg.verbose
        callbacks = CallbackList(add_history=True, add_progbar=True if verbose else False)
        cfg, callbacks = setup_callbacks(cfg, callbacks, validation)
        callbacks.set_model(model)
        callbacks.set_params(dict(verbose=verbose, epochs=cfg.epochs))

        self.callbacks = callbacks

        logs = gf.BunchDict()

        callbacks.on_train_begin()
        # for some initialization
        if hasattr(model, 'on_train_begin'):
            model.on_train_begin()

        if verbose:
            print("Training...")

        try:
            for epoch in range(cfg.epochs):
                callbacks.on_epoch_begin(epoch)
                train_logs = self.train_step(train_data)
                if hasattr(train_data, 'on_epoch_end'):
                    train_data.on_epoch_end()
                logs.update({k: to_item(v) for k, v in train_logs.items()})

                if validation:
                    valid_logs = self.test_step(val_data)
                    logs.update({("val_" + k): to_item(v) for k, v in valid_logs.items()})
                    if hasattr(val_data, 'on_epoch_end'):
                        val_data.on_epoch_end()

                callbacks.on_train_batch_end(len(train_data), logs)
                callbacks.on_epoch_end(epoch, logs)

                if model.stop_training:
                    print(f"Early Stopping at Epoch {epoch}", file=sys.stderr)
                    break

            callbacks.on_train_end()
            if ckpt_cfg.enabled:
                if ckpt_cfg.save_weights_only:
                    model.load_weights(ckpt_cfg.path)
                else:
                    self.model = model.load(ckpt_cfg.path)
        finally:
            # to avoid unexpected termination of the model
            if ckpt_cfg.enabled and ckpt_cfg.remove_weights:
                self.remove_weights()

    def evaluate(self, test_data, **kwargs):

        if not self.model:
            raise RuntimeError(
                'You must compile your model before training/testing/predicting. Use `trainer.build()`.'
            )

        cache = self.cache
        cfg = self.cfg.evaluate
        cfg.merge_from_dict(kwargs)

        if not isinstance(test_data, (DataLoader, Dataset)):
            test_data = self.test_loader(test_data)

        if cfg.cache_test_data:
            cache.test_data = test_data

        if cfg.verbose:
            print("Testing...")

        progbar = Progbar(target=len(test_data),
                          width=cfg.Progbar.width,
                          verbose=cfg.verbose)
        logs = gf.BunchDict(**self.test_step(test_data))
        logs.update({k: to_item(v) for k, v in logs.items()})
        progbar.update(len(test_data), logs.items())
        return logs

    def train_step(self, sequence):
        model = self.model
        model.reset_metrics()

        results = None
        for epoch, batch in enumerate(sequence):
            self.callbacks.on_train_batch_begin(epoch)
            inputs, labels, out_index = unravel_batch(batch)
            results = model.train_step_on_batch(x=inputs,
                                                y=labels,
                                                out_index=out_index,
                                                device=self.device)
            self.callbacks.on_train_batch_end(epoch)
        return results

    def test_step(self, sequence):
        model = self.model
        model.reset_metrics()

        results = None
        for epoch, batch in enumerate(sequence):
            inputs, labels, out_index = unravel_batch(batch)
            self.callbacks.on_test_batch_begin(epoch)
            results = model.test_step_on_batch(x=inputs,
                                               y=labels,
                                               out_index=out_index,
                                               device=self.device)
            self.callbacks.on_test_batch_end(epoch)
        return results

    def predict(self, predict_data=None, transform=None):

        if not self.model:
            raise RuntimeError(
                'You must compile your model before training/testing/predicting. Use `trainer.build()`.'
            )

        cache = self.cache
        cfg = self.cfg.predict
        cfg.transform = transform

        if not isinstance(predict_data, (DataLoader, Dataset)):
            predict_data = self.predict_loader(predict_data)

        if cfg.cache_predict_data:
            cache.predict_data = predict_data

        logits = self.predict_step(predict_data)

        self.transform.logit_transform = T = gf.get(transform)
        logits = T(logits)
        return logits.squeeze()

    def predict_step(self, sequence):
        logits = []
        model = self.model
        for epoch, batch in enumerate(sequence):
            inputs, labels, out_index = unravel_batch(batch)
            self.callbacks.on_predict_batch_begin(epoch)
            logit = model.predict_step_on_batch(x=inputs,
                                                out_index=out_index,
                                                device=self.device)

            logits.append(logit)
            self.callbacks.on_predict_batch_end(epoch)

        return np.vstack(logits)

    def train_loader(self, inputs, **kwargs):
        raise NotImplementedError

    def test_loader(self, inputs, **kwargs):
        return self.train_loader(inputs, **kwargs)

    def predict_loader(self, inputs, **kwargs):
        return self.test_loader(inputs, **kwargs)

    def _test_predict(self, index):
        logit = self.predict(index)
        predict_class = logit.argmax(1)
        labels = self.graph.node_label[index]
        return (predict_class == labels).mean()

    # def reset_weights(self):
    #     # TODO: add pytorch support
    #     """reset the model to the first time."""
    #     model = self.model
    #     if self.backup is None:
    #         raise RuntimeError(
    #             "You must store the `backup` before `reset_weights`."
    #             "`backup` will be automatically stored when the model is built."
    #         )
    #     for w, wb in zip(model.weights, self.backup):
    #         w.assign(wb)

    @ property
    def model(self):
        return self._model

    @ model.setter
    def model(self, m):
        # Back up
        # if isinstance(m, tf.keras.Model) and m.weights:
        #     self.backup = tf.identity_n(m.weights)
        # TODO assert m is None or isinstance(m, tf.keras.Model) or torch.nn.Module
        self._model = m

    # def reset_optimizer(self):
    #     # TODO: add pytorch support
    #     model = self.model
    #     if not hasattr(model, 'optimizer'):
    #         raise RuntimeError("The model has not attribute `optimizer`!")
    #     for var in model.optimizer.variables():
    #         var.assign(tf.zeros_like(var))

    def reset_lr(self, value):
        # TODO: add pytorch support
        model = self.model
        if not hasattr(model, 'optimizer'):
            raise RuntimeError("The model has not attribute `optimizer`!")
        model.optimizer.learning_rate.assign(value)

    def remove_weights(self):
        filepath = self.cfg.fit.ModelCheckpoint.path
        if self.backend == "tensorflow":
            remove_extra_tf_files(filepath)

        if osp.exists(filepath):
            os.remove(filepath)

    def help(self, return_msg=False):
        """return help message for the `trainer`"""

        msg = f"""
**************************************Help Message for {self.name}******************************************
|First, setup a graph, run `trainer.setup_graph`, the reqiured argument are:                      |
{make_docs(self.setup_graph, self.data_step)}
|Second, build your model, run `trainer.build`, the reqiured argument are:                        |
{make_docs(self.build, self.model_step)} 
|Third, train your model, run `trainer.fit`, the reqiured argument are:                           |
{make_docs(self.fit)} 
|Finall and optionally, evaluate your model, run `trainer.evaluate`, the reqiured argument are:   |
{make_docs(self.evaluate)} 
"""
        if return_msg:
            return msg
        else:
            print(msg)

#     def __getattr__(self, attr):
#         ##### FIXME: This may cause ERROR ######
#         try:
#             return self.__dict__[attr]
#         except KeyError:
#             if hasattr(self, "_model") and hasattr(self._model, attr):
#                 return getattr(self._model, attr)
#             raise AttributeError(
#                 f"'{self.name}' and '{self.name}.model' objects have no attribute '{attr}'"
#             )


def to_item(value):
    if value is None:
        return value
    elif hasattr(value, 'numpy'):
        value = value.numpy()

    if hasattr(value, 'item'):
        value = value.item()

    return value


def remove_extra_tf_files(filepath):
    # for tensorflow weights that saved without h5 formate
    for ext in (".data-00000-of-00001", ".index"):
        path = filepath + ext
        if osp.exists(path):
            os.remove(path)

    filedir = osp.split(osp.realpath(filepath))[0]

    path = osp.join(filedir, "checkpoint")
    if osp.exists(path):
        os.remove(path)


def setup_callbacks(cfg, callbacks, validation):
    ckpt_cfg = cfg.ModelCheckpoint

    if not validation:
        if ckpt_cfg.enabled and ckpt_cfg.monitor.startswith("val_"):
            ckpt_cfg.enabled = False
            ckpt_cfg.monitor = ckpt_cfg.monitor[4:]

    if ckpt_cfg.enabled:
        if not ckpt_cfg.path.endswith(gg.file_ext()):
            ckpt_cfg.path += gg.file_ext()
        makedirs_from_filepath(ckpt_cfg.path)
        mc_callback = ModelCheckpoint(ckpt_cfg.path,
                                      mode=ckpt_cfg.mode,
                                      monitor=ckpt_cfg.monitor,
                                      save_best_only=ckpt_cfg.save_best_only,
                                      save_weights_only=ckpt_cfg.save_weights_only,
                                      verbose=ckpt_cfg.verbose)
        callbacks.append(mc_callback)
    return cfg, callbacks
