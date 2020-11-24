import random
import torch
import os
import sys

import numpy as np
import tensorflow as tf
import os.path as osp
import scipy.sparse as sp

from tensorflow.keras import backend as K

from graphgallery.data.io import makedirs_from_filename
from graphgallery.utils import saver
import graphgallery as gg

from .model import Model


class GalleryModel(Model):
    """Base model for semi-supervised learning and unsupervised learning."""

    def __init__(self, *graph, device="cpu:0", seed=None, name=None, **kwargs):
        r"""Create a Base model for semi-supervised learning and unsupervised learning.

        Parameters:
        ----------
            graph: Graph or MultiGraph.
            device: string. optional
                The device where the model running on.
            seed: interger scalar. optional
                Used in combination with `tf.random.set_seed` & `np.random.seed`
                & `random.seed` to create a reproducible sequence of tensors
                across multiple calls.
            name: string. optional
                Specified name for the model. (default: :str: `class.__name__`)
            kwargs: other custom keyword parameters.

        """
        super().__init__(*graph, device=device, seed=seed, name=name, **kwargs)

        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.predict_data = None
        self.backup = None

        self._model = None
        self._custom_objects = None  # used for save/load TF model

        # checkpoint path
        # add random integer to avoid duplication
        _id = np.random.RandomState(None).randint(100)
        self.ckpt_path = osp.join(os.getcwd(), f"{self.name}_checkpoint_{_id}{gg.file_postfix()}")

    def save(self, path=None, as_model=False, overwrite=True, save_format=None, **kwargs):

        if not path:
            path = self.ckpt_path

        makedirs_from_filename(path)

        if as_model:
            if self.backend == "tensorflow":
                saver.save_tf_model(self.model, path, overwrite=overwrite, save_format=save_format, **kwargs)
            else:
                saver.save_torch_model(self.model, path, overwrite=overwrite, save_format=save_format, **kwargs)
        else:
            if self.backend == "tensorflow":
                saver.save_tf_weights(self.model, path, overwrite=overwrite, save_format=save_format)
            else:
                saver.save_torch_weights(self.model, path, overwrite=overwrite, save_format=save_format)

    def load(self, path=None, as_model=False):
        if not path:
            path = self.ckpt_path

        # if not osp.exists(path):
        #     print(f"{path} do not exists.", file=sys.stderr)
        #     return

        if as_model:
            if self.backend == "tensorflow":
                self.model = saver.load_tf_model(
                    path, custom_objects=self.custom_objects)
            else:
                self.model = saver.load_torch_model(path)
        else:
            if self.backend == "tensorflow":
                saver.load_tf_weights(self.model, path)
            else:
                saver.load_torch_weights(self.model, path)

    def __getattr__(self, attr):
        ##### FIXME: This may cause ERROR ######
        try:
            return self.__dict__[attr]
        except KeyError:
            if hasattr(self, "_model") and hasattr(self._model, attr):
                return getattr(self._model, attr)
            raise AttributeError(
                f"'{self.name}' and '{self.name}.model' objects have no attribute '{attr}'")

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, m):
        # Back up
        if isinstance(m, tf.keras.Model) and m.weights:
            self.backup = tf.identity_n(m.weights)
        # TODO assert m is None or isinstance(m, tf.keras.Model) or torch.nn.Module
        self._model = m

    @property
    def custom_objects(self):
        return self._custom_objects

    @custom_objects.setter
    def custom_objects(self, objs):
        assert isinstance(objs, dict)
        self._custom_objects = objs

    def close(self):
        """Close the session of model and empty cache."""
        gg.empty_cache()
        self.model = None

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def __repr__(self):
        return f"{self.name}(device={self.device}, backend={self.backend})"
