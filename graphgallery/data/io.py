import os
import logging
import errno
import os.path as osp
from tensorflow.keras.utils import get_file

from graphgallery import is_list_like


def download_file(raw_paths, urls):

    last_except = None
    for file_name, url in zip(raw_paths, urls):
        try:
            get_file(file_name, origin=url)
        except Exception as e:
            last_except = e

    if last_except is not None:
        raise last_except


def files_exist(files):
    if is_list_like(files):
        return len(files) != 0 and all([osp.exists(f) for f in files])
    else:
        return osp.exists(files)


def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


def makedirs_from_path(path, verbose=True):
    file_dir = osp.split(osp.realpath(path))[0]
    if not osp.exists(file_dir):
        makedirs(file_dir)
        if verbose:
            logging.log(logging.WARNING,
                        f"Make Directory in {path}.")
