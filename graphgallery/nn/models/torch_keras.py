import torch
import torch.nn as nn
import os.path as osp

import graphgallery as gg
from graphgallery import functional as gf

class TorchKeras(nn.Module):
    """Keras like PyTorch Model."""

    def __init__(self, *args, **kwargs):
        self.__doc__ = super().__doc__

        super().__init__(*args, **kwargs)

        # To be compatible with TensorFlow
        self._in_multi_worker_mode = dummy_function
        self._is_graph_network = dummy_function
        self.distribute_strategy = None

        # initialize
        self.optimizer = None
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
        
    def train_step_on_batch(self,
                            x,
                            y=None,
                            out_weight=None,
                            device="cpu"):
        self.train()
        optimizer = self.optimizer
        loss_fn = self.loss
        metrics = self.metrics
        optimizer.zero_grad()

        out = self(*x if isinstance(x, (list, tuple)) else [x])
        if out_weight is not None:
            out = out[out_weight]
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        for metric in metrics:
            metric.update_state(y.cpu(), out.detach().cpu())

        results = [loss.cpu().detach()] + [metric.result() for metric in metrics]
        return dict(zip(self.metrics_names, results))

    @torch.no_grad()
    def test_step_on_batch(self,
                           x,
                           y=None,
                           out_weight=None,
                           device="cpu"):
        self.eval()
        loss_fn = self.loss
        metrics = self.metrics

        out = self(*x if isinstance(x, (list, tuple)) else [x])
        if out_weight is not None:
            out = out[out_weight]
        loss = loss_fn(out, y)
        for metric in metrics:
            metric.update_state(y.cpu(), out.detach().cpu())

        results = [loss.cpu().detach()] + [metric.result() for metric in metrics]
        return dict(zip(self.metrics_names, results))

    @torch.no_grad()
    def predict_step_on_batch(self, x, out_weight=None, device="cpu"):
        self.eval()
        out = self(*x if isinstance(x, (list, tuple)) else [x])
        if out_weight is not None:
            out = out[out_weight]
        return out.cpu().detach()

    def build(self, inputs):
        # TODO
        pass

    def compile(self, loss=None, optimizer=None, metrics=None):
        self.loss = loss
        self.optimizer = optimizer
        if not isinstance(metrics, (list, tuple)):
            metrics = [metrics]
        self.metrics = metrics

    def reset_metrics(self):
        assert self.metrics is not None
        for metric in self.metrics:
            metric.reset_states()

    @property
    def metrics_names(self):
        assert self.metrics is not None
        return ['loss'] + [metric.name for metric in self.metrics]

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

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()


def dummy_function(*args, **kwargs):
    ...
