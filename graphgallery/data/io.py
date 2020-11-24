import os
import errno
import zipfile
import os.path as osp
import numpy as np
from tensorflow.keras.utils import get_file
from typing import List

__all__ = ['download_file', 'files_exist', 'makedirs',
           'makedirs_from_filepath',
           'extract_zip', 'clean', 'load_npz']


def download_file(raw_paths: List[str],
                  urls: List[str]) -> None:
    if isinstance(raw_paths, str):
        raw_paths = (raw_paths,)
    if isinstance(urls, str):
        urls = (urls,)

    assert len(raw_paths) == len(urls)

    exceptions = []
    for filename, url in zip(raw_paths, urls):
        if not osp.exists(filename):
            try:
                get_file(filename, origin=url, extract=False)
            except Exception as e:
                exceptions.append(e)
                print(f"Downloading failed: {url}")

    if exceptions:
        raise exceptions[0]


def extract_zip(filename, folder=None):
    r"""Extracts a zip archive to a specific folder.

    Parameters:
    -----------
    filename (string): The path to the tar archive.
    folder (string): The folder.
    """
    if folder is None:
        folder = osp.realpath(osp.expanduser(osp.dirname(filename)))

    with zipfile.ZipFile(filename, 'r') as f:
        f.extractall(folder)


def clean(filepaths):
    for path in filepaths:
        if osp.exists(path):
            os.unlink(path)


def files_exist(files: List[str]) -> bool:
    if isinstance(files, (list, tuple)):
        return len(files) != 0 and all([osp.exists(f) for f in files])
    else:
        return osp.exists(files)


def makedirs(path: str) -> None:
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)), exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


def makedirs_from_filepath(filepath: str, verbose: bool = True) -> None:
    folder = osp.realpath(osp.expanduser(osp.dirname(filepath)))
    makedirs(folder)


def load_npz(filepath):
    filepath = osp.abspath(osp.expanduser(osp.realpath(filepath)))

    if not filepath.endswith('.npz'):
        filepath = filepath + '.npz'
    if osp.isfile(filepath):
        with np.load(filepath, allow_pickle=True) as loader:
            loader = dict(loader)
            for k, v in loader.items():
                if v.dtype.kind == 'O':
                    loader[k] = v.tolist()
            return loader
    else:
        raise ValueError(f"{filepath} doesn't exist.")
