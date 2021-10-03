try:
    import tensorflow as tf
    TF_ENABLED = True
except ImportError:
    TF_ENABLED = False

from graphgallery.backend import *
from graphgallery.data_type import *

# modules
from graphgallery import nn
from graphgallery import backend
from graphgallery import gallery
from graphgallery import utils
from graphgallery import data
from graphgallery import datasets
from graphgallery import functional
# from graphgallery import attack

from .version import __version__

__all__ = ["nn", "gallery", "utils", "data", "datasets",
           "backend", "functional", "attack", "__version__"
           ]
