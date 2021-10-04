import torch
import warnings
import torch.nn as nn
import os.path as osp

import graphgallery as gg
from graphgallery import functional as gf


class TorchEngine(nn.Module):
    """High-level encapsulation of PyTorch Model."""

    def __init__(self, *args, **kwargs):
        self.__doc__ = super().__doc__

        super().__init__(*args, **kwargs)

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

    def forward_step(self, x, out_index=None):
        # the input `x` can be: (1) dict (2) list or tuple like
        if isinstance(x, dict):
            output_dict = self(**x)
        else:
            if not isinstance(x, (list, tuple)):
                x = (x,)
            output_dict = self(*x)

        if not isinstance(output_dict, dict):
            if isinstance(output_dict, tuple):
                raise RuntimeError("For model more than 1 outputs, we recommend you to use dict as returns.")
            # Here `z` is the final representation of the model
            z = output_dict
            output_dict = dict(z=z)
        else:
            z = output_dict['z']
        # index select or mask outputs
        pred = self.index_select(z, out_index=out_index)
        output_dict['pred'] = pred
        return output_dict

    def compute_loss(self, output_dict, y):
        loss = self.loss(output_dict['pred'], y)
        return loss

    def loss_backward(self, loss):
        loss.backward()

    def index_select(self, z, out_index=None):
        if isinstance(z, (list, tuple)):
            return list(self.index_select(x, out_index=out_index) for x in z)

        if out_index is None:
            return z

        if isinstance(out_index, slice) or out_index.ndim <= 1:
            pred = z[out_index]
        elif out_index.ndim == 2:
            pred = z[out_index[0], out_index[1]]
        else:
            warnings.warn(f'UNKNOWN out_index `{out_index}`', UserWarning)
        return pred

    def compute_metrics(self, output_dict, y):
        pred = output_dict['pred']
        if isinstance(y, dict):
            y = y['y']
        elif isinstance(y, (tuple, list)):
            y = y[0]

        if y is None:
            y = output_dict.get('y', None)

        assert y is not None

        for metric in self.metrics:
            metric.update_state(y.cpu(), pred.detach().cpu())
        return [metric.result() for metric in self.metrics]

    def train_step_on_batch(self,
                            x,
                            y,
                            out_index=None,
                            device="cpu"):
        self.train()
        optimizer = self.optimizer
        optimizer.zero_grad()
        x, y = to_device(x, y, device=device)
        # Step 1. forward
        output_dict = self.forward_step(x, out_index=out_index)
        # Step 2. compute loss
        loss = self.compute_loss(output_dict, y)
        # Step 3. loss backward
        self.loss_backward(loss)

        # Step 4. optimizer step
        optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        # Step 4. update evaluation metrics
        metrics = self.compute_metrics(output_dict, y)
        del output_dict

        results = [loss.cpu().item()] + metrics

        return dict(zip(self.metrics_names, results))

    @torch.no_grad()
    def test_step_on_batch(self,
                           x,
                           y,
                           out_index=None,
                           device="cpu"):

        self.eval()
        x, y = to_device(x, y, device=device)
        # Step 1. forward
        output_dict = self.forward_step(x, out_index=out_index)
        # Step 2. compute loss
        loss = self.compute_loss(output_dict, y)
        # Step 3. update evaluation metrics
        metrics = self.compute_metrics(output_dict, y)

        del output_dict
        if loss is not None:
            loss = loss.cpu().item()

        results = [loss] + metrics

        return dict(zip(self.metrics_names, results))

    @torch.no_grad()
    def predict_step_on_batch(self, x, out_index=None, device="cpu"):
        self.eval()
        x, _ = to_device(x, device=device)
        output_dict = self.forward_step(x, out_index=out_index)
        z = output_dict['pred']
        return z.detach().cpu()

    def freeze(self, module=None):
        if module is None:
            module = self
        for para in module.parameters():
            para.requires_grad = False

    def defrozen(self, module=None):
        if module is None:
            module = self
        for para in module.parameters():
            para.requires_grad = True

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
        return f"(optimizer): {self.optimizer}\n(scheduler): {self.scheduler}"


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

    def wrapper(inputs):
        # The condiction `not gg.is_scalar(inputs[0])` used to
        # avoid a python tuple (1, 2, ..., N) as inputs
        if isinstance(inputs, (list, tuple)) and not gg.is_scalar(inputs[0]):
            return tuple(wrapper(input) for input in inputs)
        elif isinstance(inputs, dict):
            for k, v in inputs.items():
                inputs[k] = wrapper(v)
            return inputs
        else:
            return inputs.to(device) if hasattr(inputs, 'to') else inputs

    return wrapper(x), wrapper(y)
