import torch
import six
import os.path as osp
import tensorflow as tf

import graphgallery as gg

from graphgallery.nn.models import TFKeras


def save_tf_model(model, filepath, overwrite=True, save_format=None, **kwargs):

    ext = gg.file_ext()
    if not filepath.endswith(ext):
        filepath = filepath + ext

    model.save(filepath, overwrite=overwrite, save_format=save_format, **kwargs)


def save_torch_model(model, filepath, overwrite=True, save_format=None, **kwargs):
    ext = gg.file_ext()

    if not filepath.endswith(ext):
        filepath = filepath + ext

    if not overwrite and osp.isfile(filepath):
        proceed = ask_to_proceed_with_overwrite(filepath)
        if not proceed:
            return

    torch.save(model, filepath)


def save_tf_weights(model, filepath, overwrite=True,
                    save_format=None, **kwargs):
    ext = gg.file_ext()

    if not filepath.endswith(ext):
        filepath = filepath + ext
    try:
        model.save_weights(filepath, overwrite=overwrite,
                           save_format=save_format, **kwargs)
    except ValueError as e:
        model.save_weights(filepath[:-3], overwrite=overwrite,
                           save_format=save_format, **kwargs)


def save_torch_weights(model, filepath, overwrite=True,
                       save_format=None, **kwargs):
    ext = gg.file_ext()

    if not filepath.endswith(ext):
        filepath = filepath + ext

    if not overwrite and osp.isfile(filepath):
        proceed = ask_to_proceed_with_overwrite(filepath)
        if not proceed:
            return

    torch.save(model.state_dict(), filepath)


def load_tf_weights(model, filepath):
    ext = gg.file_ext()

    if not filepath.endswith(ext):
        filepath = filepath + ext
    try:
        model.load_weights(filepath)
    except KeyError as e:
        model.load_weights(filepath[:-3])


def load_torch_weights(model, filepath):
    ext = gg.file_ext()

    if not filepath.endswith(ext):
        filepath = filepath + ext

    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint)


def load_tf_model(filepath, custom_objects=None, **kwargs):
    ext = gg.file_ext()

    if not filepath.endswith(ext):
        filepath = filepath + ext
        
    if custom_objects:
        custom_objects['TFKeras'] = TFKeras
        
    return tf.keras.models.load_model(filepath,
                                      custom_objects=custom_objects, **kwargs)


def load_torch_model(filepath):
    ext = gg.file_ext()

    if not filepath.endswith(ext):
        filepath = filepath + ext

    return torch.load(filepath)


def ask_to_proceed_with_overwrite(filepath):
    """Produces a prompt asking about overwriting a file.

    Parameters:
      filepath: the path to the file to be overwritten.

    Returns:
      True if we can proceed with overwrite, False otherwise.
    """
    overwrite = six.moves.input('[WARNING] %s already exists - overwrite? '
                                '[y/n]' % (filepath)).strip().lower()
    while overwrite not in ('y', 'n'):
        overwrite = six.moves.input('Enter "y" (overwrite) or "n" '
                                    '(cancel).').strip().lower()
    if overwrite == 'n':
        return False
    print('[TIP] Next time specify overwrite=True!')
    return True
