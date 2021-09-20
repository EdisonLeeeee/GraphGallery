import torch
import warnings
import torch.nn as nn
import os.path as osp

import graphgallery as gg
from graphgallery import functional as gf


class TorchKeras(nn.Module):
    """Keras like PyTorch Model."""

    def __init__(self, *args, **kwargs):
        self.__doc__ = super().__doc__

        super().__init__(*args, **kwargs)

        # # To be compatible with TensorFlow
        # self._in_multi_worker_mode = dummy_function
        # self._is_graph_network = dummy_function
        # self.distribute_strategy = None

        # # To be compatible with TensorBoard callbacks
        # self._train_counter = 0
        # self._test_counter = 0

        # initialize
        self.optimizer = None
        self.scheduler = None
        self.metrics = None
        self.loss = None

        # cache
        self.empty_cache()

    def from_cache(self, **kwargs):
        if not kwargs:
            return None

        def get(name, value):
            obj = self.cache.get(name, None)

            if obj is None:
                assert value is not None
                self.cache[name] = value
                obj = value
            return obj

        out = tuple(get(k, v) for k, v in kwargs.items())
        if len(out) == 1:
            out, = out
        return out

    def empty_cache(self):
        self.cache = gf.BunchDict()
        import gc
        gc.collect()

    def compute_loss(self, out, y):
        return self.loss(out, y)

    def index_select(self, out, out_index=None):
        if out_index is None:
            return out
        if out_index.ndim <= 1:
            out = out[out_index]
        elif out_index.ndim == 2:
            out = out[out_index[0], out_index[1]]
        else:
            warnings.warn(f'UNKNOWN out_index `{out_index}`', UserWarning)
        return out

    def update_metrics(self, out, y):
        for metric in self.metrics:
            metric.update_state(y.cpu(), out.detach().cpu())

    def train_step_on_batch(self,
                            x,
                            y,
                            out_index=None,
                            device="cpu"):
        self.train()
        optimizer = self.optimizer
        optimizer.zero_grad()
        x, y = to_device(x, y, device=device)
        # 1. forward
        out = self(*x)
        # 2. index select or mask outputs
        out = self.index_select(out, out_index=out_index)
        # 3. compute loss and update model
        loss = self.compute_loss(out, y)
        loss.backward()
        optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        # 4. update evaluation metrics
        self.update_metrics(out, y)

        results = [loss.cpu().detach()] + [metric.result() for metric in self.metrics]
        return dict(zip(self.metrics_names, results))

    @torch.no_grad()
    def test_step_on_batch(self,
                           x,
                           y,
                           out_index=None,
                           device="cpu"):
        self.eval()
        metrics = self.metrics
        x, y = to_device(x, y, device=device)
        # 1. forward
        out = self(*x)
        # 2. index select or mask outputs
        out = self.index_select(out, out_index=out_index)
        # 3. compute loss
        loss = self.compute_loss(out, y)
        # 4. update evaluation metrics
        self.update_metrics(out, y)

        if loss is not None:
            loss = loss.cpu().item()

        results = [loss] + [metric.result() for metric in metrics]
        return dict(zip(self.metrics_names, results))

    @torch.no_grad()
    def predict_step_on_batch(self, x, out_index=None, device="cpu"):
        self.eval()
        x = to_device(x, device=device)
        out = self.index_select(self(*x), out_index=out_index)
        return out.cpu().detach()

    def build(self, inputs):
        # TODO
        pass

    def compile(self, loss=None, optimizer=None, metrics=None,
                scheduler=None):
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        if not isinstance(metrics, (list, tuple)):
            metrics = [metrics]
        self.metrics = metrics

    def reset_metrics(self):
        assert self.metrics is not None
        for metric in self.metrics:
            metric.reset_states()

    def reset_parameter(self):
        reset(self)

    @property
    def metrics_names(self):
        assert self.metrics is not None
        return ['loss'] + [metric.name for metric in self.metrics]

    def on_train_begin(self):
        pass

    def on_test_begin(self):
        pass

    def on_predict_begin(self):
        pass

    def summary(self):
        # TODO
        pass

    def save_weights(self,
                     filepath,
                     overwrite=True,
                     save_format=None,
                     **kwargs):
        ext = gg.file_ext()

        if not filepath.endswith(ext):
            filepath = filepath + ext

        if not overwrite and osp.isfile(filepath):
            proceed = gg.utils.ask_to_proceed_with_overwrite(filepath)
            if not proceed:
                return

        torch.save(self.state_dict(), filepath)

    def load_weights(self, filepath):
        ext = gg.file_ext()

        if not filepath.endswith(ext):
            filepath = filepath + ext

        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint)

    def save(self, filepath, overwrite=True, save_format=None, **kwargs):
        ext = gg.file_ext()

        if not filepath.endswith(ext):
            filepath = filepath + ext

        if not overwrite and osp.isfile(filepath):
            proceed = gg.utils.ask_to_proceed_with_overwrite(filepath)
            if not proceed:
                return

        torch.save(self, filepath)

    @classmethod
    def load(cls, filepath):
        ext = gg.file_ext()

        if not filepath.endswith(ext):
            filepath = filepath + ext

        return torch.load(filepath)

    def extra_repr(self):
        return f"(optimizer): {self.optimizer}"


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


def to_device(x, y=None, device='cpu'):
    if not isinstance(x, tuple):
        x = (x,)

    def wrapper(inputs):
        # FIXME: `len(inputs) < 20` used to avoid the inputs is a python tuple (1, 2, ..., N)
        if isinstance(inputs, tuple) and len(inputs) < 20:
            return tuple(wrapper(input) for input in inputs)
        else:
            return inputs.to(device) if hasattr(inputs, 'to') else inputs
    if y is not None:
        return wrapper(x), wrapper(y)
    else:
        return wrapper(x)
