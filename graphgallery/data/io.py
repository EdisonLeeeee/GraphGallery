import io
import os
import tarfile
import errno
import json
import zipfile
import shutil
import pickle
import os.path as osp
import numpy as np
import pandas as pd

from six.moves.urllib.error import HTTPError
from six.moves.urllib.error import URLError
from six.moves.urllib.request import urlretrieve

from graphgallery.utils import Progbar

__all__ = ['load_pickle', 'dump_pickle',
           'download_file', 'files_exist',
           'makedirs', 'makedirs_from_filepath', 'makedirs_rm_exist',
           'extractall', 'remove',
           'load_npz', 'read_csv', 'read_json', 'get_file',
           ]


def load_pickle(fname):
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
    return obj


def dump_pickle(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)
    return fname


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


def extractall(filename, folder=None, archive_format='auto'):
    """Extracts an archive if it matches tar, tar.gz, tar.bz, or zip formats.

    Parameters:
    -----------
    filename: path to the archive file, which could be a list of paths.
    folder: folder to extract the archive file
    archive_format: Archive format to try for extracting the file.
        Options are 'auto', 'tar', 'zip', and None.
        'tar' includes tar, tar.gz, and tar.bz files.
        The default 'auto' is ['tar', 'zip'].
        None or an empty list will return no matches found.

    Returns:
    -----------
    True if a match was found and an archive extraction was completed,
        False otherwise.
    """

    if not filename:
        return False

    if isinstance(filename, (list, tuple)):
        for f in filename:
            extractall(f, folder)
        return True

    if folder is None:
        folder = osp.dirname(osp.realpath(osp.expanduser(filename)))

    if archive_format is None:
        return False
    if archive_format == 'auto':
        archive_format = ['tar', 'zip']
    if isinstance(archive_format, str):
        archive_format = [archive_format]

    for archive_type in archive_format:
        if archive_type == 'tar':
            open_fn = tarfile.open
            is_match_fn = tarfile.is_tarfile
        elif archive_type == 'zip':
            open_fn = zipfile.ZipFile
            is_match_fn = zipfile.is_zipfile
        else:
            raise ValueError(archive_type)

        if is_match_fn(filename):
            with open_fn(filename) as archive:
                try:
                    archive.extractall(folder)
                except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
                    if osp.exists(folder):
                        if osp.isfile(folder):
                            os.remove(folder)
                        else:
                            shutil.rmtree(folder)
                    raise
            return True
    return False


def get_file(fname,
             origin,
             untar=False,
             cache_subdir='datasets',
             extract=False,
             archive_format='auto',
             cache_dir=None):
    """Downloads a file from a URL if it not already in the cache.

    By default the file at the url `origin` is downloaded to the
    cache_dir `~/.graphgallery`, placed in the cache_subdir `datasets`,
    and given the filename `fname`. The final location of a file
    `example.txt` would therefore be `~/.graphgallery/datasets/example.txt`.

    Files in tar, tar.gz, tar.bz, and zip formats can also be extracted.
    Passing a hash will verify the file after download. The command line
    programs `shasum` and `sha256sum` can compute the hash.

    Note:
    -----------
    This function was initially written by TensorFlow Authors. See
    `tf.keras.utils.get_file`. Copyright The TensorFlow Authors. All Rights Reserved.


    Parameters:
    -----------
    fname: Name of the file. If an absolute path `/path/to/file.txt` is
        specified the file will be saved at that location.
    origin: Original URL of the file.
    untar: Deprecated in favor of 'extract'.
        boolean, whether the file should be decompressed
    cache_subdir: Subdirectory under the GraphGallery cache dir where the file is
        saved. If an absolute path `/path/to/folder` is
        specified the file will be saved at that location.
    extract: True tries extracting the file as an Archive, like tar or zip.
    archive_format: Archive format to try for extracting the file.
        Options are 'auto', 'tar', 'zip', and None.
        'tar' includes tar, tar.gz, and tar.bz files.
        The default 'auto' is ['tar', 'zip'].
        None or an empty list will return no matches found.
    cache_dir: Location to store cached files, when None it
        defaults to the [GraphGallery Directory]

    Returns:
    -----------
    Path to the downloaded file
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.graphgallery')

    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.graphgallery')
    datadir = os.path.join(datadir_base, cache_subdir)
    makedirs(datadir)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)

    if not os.path.exists(fpath):
        print('Downloading data from', origin)

        class ProgressTracker(object):
            # Maintain progbar for the lifetime of download.
            # This design was chosen for Python 2.7 compatibility.
            progbar = None

        def dl_progress(count, block_size, total_size):
            if ProgressTracker.progbar is None:
                if total_size == -1:
                    total_size = None
                ProgressTracker.progbar = Progbar(total_size)
            else:
                ProgressTracker.progbar.update(count * block_size)

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(origin, fpath, dl_progress)
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise
        ProgressTracker.progbar = None

    if untar:
        if not os.path.exists(untar_fpath):
            extractall(fpath, datadir, archive_format='tar')
        return untar_fpath

    if extract:
        extractall(fpath, datadir, archive_format)

    return fpath


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


def makedirs(folder: str):
    try:
        os.makedirs(osp.expanduser(osp.normpath(folder)), exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(folder):
            raise e


def makedirs_from_filepath(filepath: str, verbose: bool = True):
    folder = osp.dirname(osp.realpath(osp.expanduser(filepath)))
    makedirs(folder)


def makedirs_rm_exist(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)


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


def read_csv(reader, dtype=np.int32, sep=","):
    if isinstance(reader, str):
        reader = osp.abspath(osp.expanduser(reader))
    else:
        reader = io.BytesIO(reader)
    return pd.read_csv(reader,
                       encoding="utf8",
                       sep=sep,
                       dtype={"switch": dtype})


def read_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
