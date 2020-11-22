import os
import errno
import os.path as osp
from tensorflow.keras.utils import get_file
from typing import List


def download_file(raw_paths: List[str],
                  urls: List[str]) -> None:

    exceptions = []
    for filename, url in zip(raw_paths, urls):
        if not osp.exists(filename):
            try:
                get_file(filename, origin=url)
            except Exception as e:
                exceptions.append(e)
                print(f"Downloading failed: {url}")

    if exceptions:
        raise exceptions[0]


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


def makedirs_from_filename(filename: str, verbose: bool = True) -> None:
    folder = osp.realpath(osp.expanduser(osp.dirname(filename)))
    makedirs(folder)
