import os.path as osp
import pickle as pkl

from typing import Optional, List

from .dataset import Dataset
from ..data.io import makedirs, files_exist, download_file, extract_zip, remove


class InMemoryDataset(Dataset):
    r"""Dataset base class for creating graph datasets which fit completely
    into CPU memory.
    motivated by pytorch_geometric <https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/data/in_memory_dataset.py>
    """

    def __init__(self,
                 name=None,
                 root=None,
                 url=None,
                 transform=None,
                 verbose=True,
                 remove_download=False):

        super().__init__(name=name, root=root,
                         transform=transform,
                         verbose=verbose, url=url)
        self.remove_download = remove_download
        self.download()
        self.process()

    def download(self):

        if files_exist(self.raw_paths) or files_exist(self.process_path):
            if self.verbose:
                print(f"Dataset '{self.name}' has already existed, loading it...")
            return
        elif files_exist(self.download_paths):
            extract_zip(self.download_paths)
            if self.verbose:
                print(
                    f"Dataset '{self.name}' has already existed, extracting it..."
                )
            return

        if self.verbose:
            print("Downloading...")

        self._download()

        if self.verbose:
            print("Downloading completed.")

    def _download(self):
        makedirs(self.download_dir)
        download_file(self.download_paths, self.urls)
        extract_zip(self.download_paths)

        if self.remove_download:
            remove(self.download_paths)

    def process(self):

        if files_exist(self.process_path):
            if self.verbose:
                print(f"Processed dataset '{self.name}' has already existed, loading it...")
            with open(self.process_path, 'rb') as f:
                cache = pkl.load(f)
        else:
            if self.verbose:
                print(f"Processing dataset '{self.name}'...")
            cache = self._process()
            if self.verbose:
                print("Processing completed.")

        self._graph = cache.pop('graph')
        self.split_cache = cache

    def _process(self):
        raise NotImplementedError

    @property
    def url(self):
        return self._url

    @property
    def urls(self):
        return [self.url]

    @property
    def download_dir(self):
        return osp.join(self.root, self.name)

    @property
    def download_paths(self):
        return self.raw_paths

    @property
    def process_dir(self):
        return self.download_dir

    @property
    def process_filename(self):
        return None

    @property
    def process_path(self):
        process_filename = self.process_filename
        if process_filename:
            return osp.join(self.process_dir, process_filename)
        else:
            return None

    @property
    def raw_paths(self):
        raise NotImplementedError

    @property
    def raw_filenames(self):
        raise NotImplementedError
