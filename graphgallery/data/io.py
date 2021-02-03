import io
import os
import tarfile
import errno
import zipfile
import os.path as osp
import numpy as np
import pandas as pd

from tensorflow.keras.utils import get_file


__all__ = [
    'download_file', 'files_exist', 'makedirs', 'makedirs_from_filepath',
    'extractall', 'remove', 'load_npz', 'read_csv',
]


def download_file(raw_paths, urls):
    if isinstance(raw_paths, str):
        raw_paths = (raw_paths, )
    if isinstance(urls, str):
        urls = (urls, )

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


def extractall(filename, folder=None):
    """Extracts a zip or tar.gz (tgz) archive to a specific folder.

    Parameters:
    -----------
    filename (string): The path to the tar archive.
    folder (string): The folder.
    """
    if not filename:
        return

    if folder is None:
        folder = osp.dirname(osp.realpath(osp.expanduser(filename)))

    if isinstance(filename, (list, tuple)):
        for f in filename:
            extractall(f, folder)
        return

    if filename.endswith(".zip"):
        with zipfile.ZipFile(filename, 'r') as f:
            f.extractall(folder)

    if filename.endswith(".tgz") or filename.endswith(".tar.gz"):
        tar = tarfile.open(filename, "r:gz")
        tar.extractall(path=folder)
        tar.close()


def remove(filepaths):
    if isinstance(filepaths, str):
        filepaths = (filepaths, )
    for path in filepaths:
        if osp.exists(path):
            os.unlink(path)


def files_exist(files) -> bool:
    if not files:
        return False
    if isinstance(files, (list, tuple)):
        return len(files) != 0 and all([osp.exists(f) for f in files])
    else:
        return osp.exists(files)


def makedirs(path: str):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)), exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


def makedirs_from_filepath(filepath: str, verbose: bool = True):
    folder = osp.dirname(osp.realpath(osp.expanduser(filepath)))
    makedirs(folder)


def load_npz(filepath):
    filepath = osp.abspath(osp.expanduser(filepath))

    if not filepath.endswith('.npz'):
        filepath = filepath + '.npz'
    if osp.isfile(filepath):
        with np.load(filepath, allow_pickle=True) as loader:
            loader = dict(loader)
            for k, v in loader.items():
                if v.dtype.kind in {'O', 'U'}:
                    loader[k] = v.tolist()
            return loader
    else:
        raise ValueError(f"{filepath} doesn't exist.")


def read_csv(reader, dtype=np.int32):
    if isinstance(reader, str):
        reader = osp.abspath(osp.expanduser(reader))
    else:
        reader = io.BytesIO(reader)
    return pd.read_csv(reader,
                       encoding="utf8",
                       sep=",",
                       dtype={"switch": dtype})
