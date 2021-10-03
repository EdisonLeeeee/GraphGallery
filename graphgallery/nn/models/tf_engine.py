import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.activations import softmax

import graphgallery as gg
from graphgallery.functional.tensor.tensorflow import gather

from distutils.version import LooseVersion

if LooseVersion(tf.__version__) >= LooseVersion("2.2.0"):
    METRICS = "compiled_metrics"
    LOSS = "compiled_loss"
else:
    METRICS = "metrics"
    LOSS = "loss"


class TFEngine(Model):
    """High-level encapsulation of Tensorflow Keras Model."""
    _use_tfn = False
    _custom_objects = None

    def use_tfn(self):
        assert not self._use_tfn, "'tf.function' has been used."
        self.train_step_on_batch = tf.function(self.train_step_on_batch,
                                               experimental_relax_shapes=True)
        self.test_step_on_batch = tf.function(self.test_step_on_batch,
                                              experimental_relax_shapes=True)
        self.predict_step_on_batch = tf.function(
            self.predict_step_on_batch, experimental_relax_shapes=True)
        self._use_tfn = True

    def train_step_on_batch(self, x, y=None, out_index=None, device="CPU"):
        # FIXME: self.metrics would return '[]' for tensorflow>=2.2.0
        # See <https://github.com/tensorflow/tensorflow/issues/37990>
        # the loss or metrics must be called to build the compiled_loss
        # or compiled_metrics
        loss_fn = getattr(self, LOSS)
        metrics = getattr(self, METRICS)
        optimizer = self.optimizer

        with tf.device(device):
            with tf.GradientTape() as tape:
                out = self(x, training=True)
                out = gather(out, out_index)
                loss = loss_fn(y, out) + tf.reduce_sum(self.losses)
                if isinstance(metrics, list):
                    for metric in metrics:
                        metric.update_state(y, out)
                else:
                    metrics.update_state(y, out)

            grad = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(grad, self.trainable_variables))

            results = [loss] + [
                metric.result()
                for metric in getattr(metrics, "metrics", metrics)
            ]
            return dict(zip(self.metrics_names, results))

    def test_step_on_batch(self, x, y=None, out_index=None, device="CPU"):
        loss_fn = getattr(self, LOSS)
        metrics = getattr(self, METRICS)

        with tf.device(device):
            out = self(x, training=False)
            out = gather(out, out_index)
            loss = loss_fn(y, out) + tf.reduce_sum(self.losses)
            if isinstance(metrics, list):
                for metric in metrics:
                    metric.update_state(y, out)
            else:
                metrics.update_state(y, out)

            results = [loss] + [
                metric.result()
                for metric in getattr(metrics, "metrics", metrics)
            ]
            return dict(zip(self.metrics_names, results))

    def predict_step_on_batch(self,
                              x,
                              out_index=None,
                              return_logits=True,
                              device="CPU"):
        with tf.device(device):
            out = self(x, training=False)
            out = gather(out, out_index)
            if not return_logits:
                out = softmax(out)
        return out

    def on_train_begin(self):
        pass

    def on_test_begin(self):
        pass

    def on_predict_begin(self):
        pass

    def save_weights(self,
                     filepath,
                     overwrite=True,
                     save_format=None,
                     **kwargs):
        ext = gg.file_ext()

        if not filepath.endswith(ext):
            filepath = filepath + ext
        try:
            super().save_weights(filepath,
                                 overwrite=overwrite,
                                 save_format=save_format,
                                 **kwargs)
        except ValueError as e:
            super().save_weights(filepath[:-len(ext)],
                                 overwrite=overwrite,
                                 save_format=save_format,
                                 **kwargs)

    def load_weights(self, filepath):
        ext = gg.file_ext()

        if not filepath.endswith(ext):
            filepath = filepath + ext
        try:
            super().load_weights(filepath)
        except KeyError as e:
            super().load_weights(filepath[:-len(ext)])

    def save(self, filepath, overwrite=True, save_format=None, **kwargs):

        ext = gg.file_ext()
        if not filepath.endswith(ext):
            filepath = filepath + ext

        super().save(filepath,
                     overwrite=overwrite,
                     save_format=save_format,
                     **kwargs)

    def load(self, filepath, custom_objects=None, **kwargs):
        ext = gg.file_ext()

        if not filepath.endswith(ext):
            filepath = filepath + ext

        # if self.custom_objects:
        #     self.custom_objects['TFEngine'] = TFEngine

        return tf.keras.models.load_model(filepath,
                                          custom_objects=self.custom_objects,
                                          **kwargs)

    @property
    def custom_objects(self):
        return self._custom_objects

    @custom_objects.setter
    def custom_objects(self, objs):
        assert isinstance(objs, dict), objs
        self._custom_objects = objs
