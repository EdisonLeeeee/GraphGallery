import tensorflow as tf
from tensorflow.keras import Model


class TFKeras(Model):
    """High-level encapsulation of Tensorflow Keras Model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
