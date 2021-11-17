import sys
import inspect
import numpy as np

import torch
import torch.nn as nn

from torch import Tensor
from typing import Optional, Union, Any, Callable, List

from graphgallery.gallery.callbacks import Callback, CallbackList
from torch.utils.data import DataLoader, Dataset

import graphgallery as gg
from graphgallery import functional as gf
from graphgallery.utils import Progbar
from graphgallery.nn.metrics import Accuracy


def format_doc(d):
    msg = ""
    for i, (k, v) in enumerate(d.items()):
        if v != "UNSPECIDIED":
            msg += f"({i + 1}) `{k}`, Default is `{v}` \n"
        else:
            msg += f"({i + 1}) `{k}`, UNSPECIDIED position argument\n"
    return msg


def get_doc_dict(func):
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
        d.update(get_doc_dict(f))
    return format_doc(d)


class Trainer:
    def __init__(self, *, device="cpu", seed=None, name=None, **cfg):
        """
        Parameters:
        ----------
        device: string. optional
            The device where the model running on.
        seed: interger scalar. optional
            Used to create a reproducible sequence of tensors
            across multiple calls.
        name: string. optional
            Specified name for the model. (default: :str: `class name`)
        cfg: other custom keyword arguments. 
        """

        gg.set_seed(seed)
        self.cfg = gf.BunchDict(cfg)
        if self.cfg:
            print(f"Receiving configs:\n{self.cfg}")
        self.device = torch.device(device)
        self.data_device = self.device
        self.backend = gg.backend()

        self.seed = seed
        self.name = name or self.__class__.__name__

        self._model = None
        self._graph = None
        self._cache = gf.BunchDict()
        self.transform = gf.BunchDict()

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
        self.cache_clear()

        self.graph = gf.get(graph_transform)(graph)
        if device is not None:
            self.data_device = gf.device(device, self.backend)
        else:
            self.data_device = self.device
        _, kwargs = gf.wrapper(self.data_step)(**kwargs)
        kwargs['graph_transform'] = graph_transform

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
            the model training and testing with `tf.function` (See `graphgallery.nn.modes.tf_engine.TFEngine`).
            By default, it was `True`, which can accelerate the training and inference, by it may cause
            several errors.
        other arguments (if have) will be passed into your method 'model_step'.
        """
        if self._graph is None:
            raise RuntimeError("Please call 'trainer.setup_graph(graph)' first.")

        model, kwargs = gf.wrapper(self.model_step)(**kwargs)
        self._model = model.to(self.device)

        self.optimizer = self.config_optimizer()
        self.loss = self.config_loss()
        metrics = self.config_metrics()

        if not isinstance(metrics, list):
            metrics = [metrics]

        self.metrics = metrics
        return self

    def model_step(self, *args, **kwargs):
        """Implement you model building function here"""
        raise NotImplementedError

    def fit(self, train_data, val_data=None, epochs=100, callbacks=None, verbose=None):

        model = self.model

        if model is None:
            raise RuntimeError(
                'You must compile your model before training/testing/predicting. Use `trainer.build()`.'
            )

        if not isinstance(train_data, (DataLoader, Dataset)):
            train_data = self.config_train_data(train_data)

        validation = val_data is not None

        if validation:
            if not isinstance(val_data, (DataLoader, Dataset)):
                val_data = self.config_test_data(val_data)

        # Setup callbacks
        self.callbacks = callbacks = self.config_callbacks(verbose, epochs, callbacks=callbacks)

        logs = gf.BunchDict()
        model.stop_training = False

        if verbose:
            print("Training...")

        callbacks.on_train_begin()
        try:
            for epoch in range(epochs):
                callbacks.on_epoch_begin(epoch)
                train_logs = self.train_step(train_data)
                logs.update({k: self.to_item(v) for k, v in train_logs.items()})

                if validation:
                    valid_logs = self.test_step(val_data)
                    logs.update({("val_" + k): self.to_item(v) for k, v in valid_logs.items()})

                callbacks.on_train_batch_end(len(train_data), logs)
                callbacks.on_epoch_end(epoch, logs)

                if model.stop_training:
                    print(f"Early Stopping at Epoch {epoch}", file=sys.stderr)
                    break

        finally:
            callbacks.on_train_end()

        return self

    def train_step(self, dataloader: DataLoader) -> dict:
        """One-step training on the input dataloader.

        Parameters
        ----------
        dataloader : DataLoader
            the trianing dataloader

        Returns
        -------
        dict
            the output logs, including `loss` and `val_accuracy`, etc.
        """
        optimizer = self.optimizer
        loss_fn = self.loss
        model = self.model

        optimizer.zero_grad()
        self.reset_metrics()
        model.train()

        for epoch, batch in enumerate(dataloader):
            self.callbacks.on_train_batch_begin(epoch)
            x, y, out_index = self.unravel_batch(batch)
            x = self.to_device(x)
            y = self.to_device(y)
            if not isinstance(x, tuple):
                x = x,
            out = model(*x)
            if out_index is not None:
                out = out[out_index]
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            for metric in self.metrics:
                metric.update_state(y.cpu(), out.detach().cpu())
            self.callbacks.on_train_batch_end(epoch)

        metrics = [metric.result() for metric in self.metrics]
        results = [loss.cpu().item()] + metrics

        return dict(zip(self.metrics_names, results))

    def evaluate(self, test_data, verbose=1):

        if not self.model:
            raise RuntimeError(
                'You must compile your model before training/testing/predicting. Use `trainer.build()`.'
            )

        if not isinstance(test_data, (DataLoader, Dataset)):
            test_data = self.config_test_data(test_data)

        if verbose:
            print("Testing...")

        progbar = Progbar(target=len(test_data),
                          verbose=verbose)
        logs = gf.BunchDict(**self.test_step(test_data))
        logs.update({k: self.to_item(v) for k, v in logs.items()})
        progbar.update(len(test_data), logs)
        return logs

    @torch.no_grad()
    def test_step(self, dataloader: DataLoader) -> dict:
        loss_fn = self.loss
        model = self.model
        model.eval()
        callbacks = self.callbacks
        self.reset_metrics()

        for epoch, batch in enumerate(dataloader):
            callbacks.on_test_batch_begin(epoch)
            x, y, out_index = self.unravel_batch(batch)
            x = self.to_device(x)
            y = self.to_device(y)
            if not isinstance(x, tuple):
                x = x,
            out = model(*x)
            if out_index is not None:
                out = out[out_index]
            loss = loss_fn(out, y)
            for metric in self.metrics:
                metric.update_state(y.cpu(), out.detach().cpu())
            callbacks.on_test_batch_end(epoch)

        metrics = [metric.result() for metric in self.metrics]
        results = [loss.cpu().item()] + metrics

        return dict(zip(self.metrics_names, results))

    def predict(self, predict_data=None,
                transform: Callable = torch.nn.Softmax(dim=-1)):

        if not self.model:
            raise RuntimeError(
                'You must compile your model before training/testing/predicting. Use `trainer.build()`.'
            )

        if not isinstance(predict_data, (DataLoader, Dataset)):
            predict_data = self.predict_loader(predict_data)

        out = self.predict_step(predict_data).squeeze()
        if transform is not None:
            out = transform(out)
        return out

    @torch.no_grad()
    def predict_step(self, dataloader: DataLoader) -> Tensor:
        model = self.model
        model.eval()
        outs = []
        callbacks = self.callbacks
        for epoch, batch in enumerate(dataloader):
            callbacks.on_predict_batch_begin(epoch)
            x, y, out_index = self.unravel_batch(batch)
            x = self.to_device(x)
            if not isinstance(x, tuple):
                x = x,
            out = model(*x)
            if out_index is not None:
                out = out[out_index]
            outs.append(out)
            callbacks.on_predict_batch_end(epoch)

        return torch.cat(outs, dim=0)

    def config_optimizer(self) -> torch.optim.Optimizer:
        lr = self.cfg.get('lr', 0.01)
        weight_decay = self.cfg.get('weight_decay', 5e-4)
        return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def config_loss(self) -> Callable:
        return torch.nn.CrossEntropyLoss()

    def config_metrics(self) -> Callable:
        return Accuracy()

    def config_callbacks(self, verbose, epochs, callbacks=None) -> Callback:
        callbacks = CallbackList(callbacks=callbacks, add_history=True, add_progbar=True if verbose else False)
        callbacks.set_model(self.model)
        callbacks.set_params(dict(verbose=verbose, epochs=epochs))
        return callbacks

    def config_train_data(self, inputs, **kwargs):
        raise NotImplementedError

    def config_test_data(self, inputs, **kwargs):
        return self.config_train_data(inputs, **kwargs)

    def predict_loader(self, inputs, **kwargs):
        return self.config_test_data(inputs, **kwargs)

    @staticmethod
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

    @staticmethod
    def to_item(value: Any) -> Any:
        """Transform value to Python object

        Parameters
        ----------
        value : Any
            Tensor or Numpy array

        Returns
        -------
        Any
            Python object

        Example
        -------
        >>> x = torch.tensor(1.)
        >>> to_item(x)
        1.
        """
        if value is None:
            return value

        elif hasattr(value, 'numpy'):
            value = value.numpy()

        if hasattr(value, 'item'):
            value = value.item()

        return value

    def to_device(self, x: Any) -> Any:
        """Put `x` into the device `self.device`.

        Parameters
        ----------
        x : any object, probably `torch.Tensor`.
            the input variable used for model.

        Returns
        -------
        x : any object, probably `torch.Tensor`.
            the input variable that in the device `self.device`.
        """
        device = self.device

        def wrapper(inputs):
            if isinstance(inputs, tuple):
                return tuple(wrapper(input) for input in inputs)
            elif isinstance(inputs, dict):
                for k, v in inputs.items():
                    inputs[k] = wrapper(v)
                return inputs
            else:
                return inputs.to(device) if hasattr(inputs, 'to') else inputs

        return wrapper(x)

    def cache_clear(self):
        if hasattr(self.model, 'cache_clear'):
            self.model.cache_clear()

        self._cache = gf.BunchDict()
        import gc
        gc.collect()
        return self

    @property
    def graph(self):
        graph = self._graph
        if graph is None:
            raise KeyError("You haven't pass any graph instance.")
        return graph

    @graph.setter
    def graph(self, graph):
        assert graph is None or isinstance(graph, gg.data.BaseGraph)
        if graph is not None:
            self._graph = graph.copy()

    @property
    def cache(self):
        return self._cache

    def register_cache(self, **kwargs):
        self._cache.update(kwargs)

    @property
    def metrics_names(self) -> List[str]:
        assert self.metrics is not None
        return ['loss'] + [metric.name for metric in self.metrics]

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, m):
        assert m is None or isinstance(m, torch.nn.Module)
        self._model = m

    def reset_metrics(self):
        if self.metrics is None:
            return
        for metric in self.metrics:
            metric.reset_states()

    def __repr__(self):
        return f"{self.name}(device={self.device}, backend={self.backend})"

    __str__ = __repr__

    def _test_predict(self, index):
        logit = self.predict(index)
        predict_class = logit.argmax(1)
        labels = self.graph.label[index]
        return (predict_class == labels).mean()

    def help(self, return_msg=False):
        """return help message for the `trainer`"""

        msg = f"""
**************************************Help Message for {self.name}******************************************
|First, initialize a trainer object, run `trainer={self.name}(device='cpu', seed=42)                  |
------------------------------------------------------------------------------------------------------------
|Second, setup a graph, run `trainer.setup_graph()`, the reqiured argument are:                      |
{make_docs(self.setup_graph, self.data_step)}
------------------------------------------------------------------------------------------------------------
|Third, build your model, run `trainer.build()`, the reqiured argument are:                          |
{make_docs(self.build, self.model_step)} 
------------------------------------------------------------------------------------------------------------
|Fourth, train your model, run `trainer.fit()`, the reqiured argument are:                           |
{make_docs(self.fit)} 
------------------------------------------------------------------------------------------------------------
|Finally and optionally, evaluate your model, run `trainer.evaluate()`, the reqiured argument are:   |
{make_docs(self.evaluate)} 
------------------------------------------------------------------------------------------------------------
"""
        if return_msg:
            return msg
        else:
            print(msg)
