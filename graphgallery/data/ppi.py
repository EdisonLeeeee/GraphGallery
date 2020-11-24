import os
import json
import os.path as osp
import numpy as np
import networkx as nx
import pickle as pkl

from itertools import product
from typing import Optional, List, Tuple, Callable, Union

from graphgallery.data.dataset import Dataset
from graphgallery.data.io import makedirs, files_exist, download_file, extract_zip, clean, load_npz
from graphgallery.data.preprocess import process_planetoid_datasets
from graphgallery.data.multigraph import MultiGraph
import graphgallery.functional as F


TrainValTest = Tuple[np.ndarray]
Transform = Union[List, Tuple, str, List, Tuple, Callable]


class PPI(Dataset):
    r"""The protein-protein interaction networks from the `"Predicting
    Multicellular Function through Multi-layer Tissue Networks"
    <https://arxiv.org/abs/1707.04638>`_ paper, containing positional gene
    sets, motif gene sets and immunological signatures as features (50 in
    total) and gene ontology sets as labels (121 in total).

    The original url is: <https://data.dgl.ai/dataset/ppi.zip>
    """

#     github_url = "https://raw.githubusercontent.com/EdisonLeeeee/"
#     "GraphData/master/datasets/ppi/ppi.zip"
    github_url = 'https://data.dgl.ai/dataset/ppi.zip'

    def __init__(self, root: Optional[str] = None,
                 transform: Optional[Transform] = None,
                 verbose: bool = True):
        name = "ppi"
        super().__init__(name, root, transform, verbose)

        self.download_dir = osp.join(self.root, name)
        self.process_dir = osp.join(self.root, name)
        makedirs(self.download_dir)
        makedirs(self.process_dir)

        self.download()
        self.process()

    def download(self) -> None:

        if files_exist(self.raw_paths):
            if self.verbose:
                print(f"Dataset {self.name} have already existed.")
                self.show(self.raw_paths)
            return

        if self.verbose:
            print("Downloading...")

        download_file(self.download_paths, self.urls)
        extract_zip(self.download_paths[0])
        clean(self.download_paths)

        if self.verbose:
            self.show(self.raw_paths)
            print("Downloading completed.")

    def process(self) -> None:

        if files_exist(self.processed_paths):
            if self.verbose:
                print(f"Processed dataset {self.name} have already existed.")
                self.show(self.processed_paths)
            with open(self.processed_paths[0], 'rb') as f:
                loader = pkl.load(f)
            graph = loader.pop('graph')
        else:
            if self.verbose:
                print("Processing...")
            graph, loader = self._process()
            if self.verbose:
                print("Processing completed.")

        self.graph = self.transform(graph)
        self.split_cache = loader

    def _process(self) -> None:

        adj_matrices = []
        node_attrs = []
        node_labels = []
        graph_labels = []
        path = self.download_dir
        loader = {}
        last = 0
        for split in ("train", "valid", "test"):
            idx = np.load(os.path.join(path, f"{split}_graph_id.npy"))
            x = np.load(os.path.join(path, f"{split}_feats.npy"))
            y = np.load(os.path.join(path, f"{split}_labels.npy"))
            nx_graph_path = os.path.join(path, f"{split}_graph.json")

            with open(nx_graph_path, "r", encoding="utf-8") as f:
                G = nx.DiGraph(nx.json_graph.node_link_graph(json.load(f)))

            G = F.nx_graph_to_sparse_adj(G)
            idx = idx - idx.min()
            for i in range(idx.max() + 1):
                mask = idx == i
                adj_matrices.append(G[mask][:, mask])
                node_attrs.append(x[mask])
                node_labels.append(y[mask])
                graph_labels.append(i)

            now = len(adj_matrices)
            loader[split] = slice(last, now)
            last = now

        graph = MultiGraph(adj_matrices, node_attrs, node_labels, graph_label=graph_labels)
        with open(self.processed_paths[0], 'wb') as f:
            pkl.dump(dict(**loader, graph=graph), f)
        return graph, loader

    def split_graphs(self, train_size=None,
                     val_size=None,
                     test_size=None,
                     random_state: Optional[int] = None):
        loader = self.split_cache
        graph = self.graph
        self.splits.update(dict(train_graphs=graph[loader['train']],
                                val_graphs=graph[loader['valid']],
                                test_graphs=graph[loader['test']]))
        return self.splits

    @property
    def url(self) -> List[str]:
        return self.github_url

    @property
    def processed_filenames(self):
        return [f'{self.name}.pkl']

    @property
    def processed_paths(self) -> List[str]:
        return [osp.join(self.process_dir, process_filename) for process_filename in self.processed_filenames]

    @property
    def raw_filenames(self) -> List[str]:
        splits = ['train', 'valid', 'test']
        files = ['feats.npy', 'graph_id.npy', 'graph.json', 'labels.npy']
        return ['{}_{}'.format(s, f) for s, f in product(splits, files)]

    @property
    def download_paths(self):
        return [osp.join(self.download_dir, self.name + '.zip')]

    @property
    def raw_paths(self) -> List[str]:
        return [osp.join(self.download_dir, raw_filename) for raw_filename in self.raw_filenames]
