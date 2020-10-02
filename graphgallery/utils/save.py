import torch
import six
import os.path as osp
import tensorflow as tf

from graphgallery import POSTFIX

def save_tf_model(model, filepath, overwrite=True, save_format=None, **kwargs):

    postfix = POSTFIX
    filepath_withPOSTFIX = filepath
    if not filepath_withPOSTFIX.endswith(postfix):
        filepath_withPOSTFIX = filepath_withPOSTFIX + postfix

    model.save(filepath_withPOSTFIX, overwrite=overwrite, save_format=save_format, **kwargs)


def save_torch_model(model, filepath, overwrite=True, save_format=None, **kwargs):
    postfix = POSTFIX

    filepath_withPOSTFIX = filepath
    if not filepath_withPOSTFIX.endswith(postfix):
        filepath_withPOSTFIX = filepath_withPOSTFIX + postfix

    if not overwrite and osp.isfile(filepath_withPOSTFIX):
        proceed = ask_to_proceed_with_overwrite(filepath_withPOSTFIX)
        if not proceed:
            return

    torch.save(model, filepath_withPOSTFIX)


def save_tf_weights(model, filepath, overwrite=True, save_format=None):
    postfix = POSTFIX

    filepath_withPOSTFIX = filepath
    if not filepath_withPOSTFIX.endswith(postfix):
        filepath_withPOSTFIX = filepath_withPOSTFIX + postfix
    try:
        model.save_weights(filepath_withPOSTFIX, overwrite=overwrite, save_format=save_format)
    except ValueError as e:
        model.save_weights(filepath_withPOSTFIX[:-3], overwrite=overwrite, save_format=save_format)


def save_torch_weights(model, filepath, overwrite=True, save_format=None):
    postfix = POSTFIX

    filepath_withPOSTFIX = filepath
    if not filepath_withPOSTFIX.endswith(postfix):
        filepath_withPOSTFIX = filepath_withPOSTFIX + postfix

    if not overwrite and osp.isfile(filepath_withPOSTFIX):
        proceed = ask_to_proceed_with_overwrite(filepath_withPOSTFIX)
        if not proceed:
            return

    torch.save(model.state_dict(), filepath_withPOSTFIX)


def load_tf_weights(model, filepath):
    postfix = POSTFIX

    filepath_withPOSTFIX = filepath
    if not filepath_withPOSTFIX.endswith(postfix):
        filepath_withPOSTFIX = filepath_withPOSTFIX + postfix
    try:
        model.load_weights(filepath_withPOSTFIX)
    except KeyError as e:
        model.load_weights(filepath_withPOSTFIX[:-3])


def load_torch_weights(model, filepath):
    postfix = POSTFIX

    filepath_withPOSTFIX = filepath
    if not filepath_withPOSTFIX.endswith(postfix):
        filepath_withPOSTFIX = filepath_withPOSTFIX + postfix

    checkpoint = torch.load(filepath_withPOSTFIX)
    model.load_state_dict(checkpoint)


def load_tf_model(filepath, custom_objects=None, **kwargs):
    postfix = POSTFIX

    filepath_withPOSTFIX = filepath
    if not filepath_withPOSTFIX.endswith(postfix):
        filepath_withPOSTFIX = filepath_withPOSTFIX + postfix

    return tf.keras.models.load_model(filepath_withPOSTFIX,
                                      custom_objects=custom_objects, **kwargs)


def load_torch_model(filepath):
    postfix = POSTFIX

    filepath_withPOSTFIX = filepath
    if not filepath_withPOSTFIX.endswith(postfix):
        filepath_withPOSTFIX = filepath_withPOSTFIX + postfix

    return torch.load(filepath_withPOSTFIX)


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
