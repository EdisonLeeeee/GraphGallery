import os.path as osp
import pickle as pkl

from sklearn.utils.validation import _ensure_no_complex_data
from graphgallery import functional as gf
from sklearn.preprocessing import LabelEncoder

from ..data import Reader
from ..data.graph import Graph
from .in_memory_dataset import InMemoryDataset

_DATASET = gf.BunchDict(chameleon="Wiki-chameleon dataset (classification)",
                        crocodile="Wiki-crocodile dataset (classification)",
                        squirrel="Wiki-squirrel dataset (node classification)",
                        facebook="facebook dataset (classification)",
                        github="github dataset, the same as KarateClub('github') (classification)",
                        DE="Twitch-DE dataset (classification, regression)",
                        ENGB="Twitch-ENGB dataset (classification, regression)",
                        ES="Twitch-ES dataset (classification, regression)",
                        FR="Twitch-FR dataset (classification, regression)",
                        PTBR="Twitch-PTBR dataset (classification, regression)",
                        RU="Twitch-RU dataset (classification, regression)",
                        ZHTW="Twitch-ZHTW dataset (classification, regression)",
                        )


class MUSAE(InMemoryDataset):
    """Datasets from `MUSAE: Multi-Scale Attributed Node Embedding`, CIKM 2020
    <https://github.com/benedekrozemberczki/MUSAE/>
    """

    __url__ = "https://github.com/EdisonLeeeee/GraphData/raw/master/datasets/musae"

    def __init__(self,
                 name,
                 root=None,
                 *, attenuated=False,
                 transform=None,
                 verbose=True,
                 url=None,
                 remove_download=False):
        if not name in self.available_datasets():
            raise ValueError(
                f"Currently only support for these datasets {tuple(self.available_datasets().keys())}."
            )
        self.attenuated = attenuated
        super().__init__(name=name, root=root,
                         transform=transform,
                         verbose=verbose, url=url,
                         remove_download=remove_download)

    @staticmethod
    def available_datasets():
        return _DATASET

    @property
    def urls(self):
        return [f"{self.__url__}/{self.name}/{raw_filename}"
                for raw_filename in self.raw_filenames]

    def __process__(self):
        reader = Reader()
        filenames = self.raw_paths
        if self.attenuated:
            src = "x1"
            dst = "x2"
        else:
            if self.name in ["chameleon", "crocodile", "squirrel"]:
                src = "id1"
                dst = "id2"
            else:
                src = "id_1"
                dst = "id_2"

        try:
            adj_matrix = reader.read_edges(filenames[0], src=src, dst=dst)
        except KeyError:
            adj_matrix = reader.read_edges(filenames[0], src="from", dst="to")

        node_attr = reader.read_json_features(filenames[1])
        data = reader.read_target(filenames[2], return_target=False)

        if self.name == "facebook":
            node_label, node_class_name = encode(data, "page_type")
            node_name = data['page_name'].to_numpy()
            graph = Graph(adj_matrix, node_attr, node_label, copy=False,
                          metadata={"node_class_name": node_class_name,
                                    "node_name": node_name})

        elif self.name in ["chameleon", "crocodile", "squirrel"]:
            node_label = reader.read_target(filenames[2])
            graph = Graph(adj_matrix, node_attr, node_label, copy=False)
        elif self.name == "github":
            node_label = data['ml_target'].to_numpy()
            node_name = data['name'].to_numpy()
            graph = Graph(adj_matrix, node_attr, node_label, copy=False,
                          metadata={"node_name": node_name})
        else:
            if 'id' in data.columns:
                del data['id']
            data["mature"] = LabelEncoder().fit_transform(data["mature"])
            data["partner"] = LabelEncoder().fit_transform(data["partner"])
            node_label = data.to_numpy()
            graph = Graph(adj_matrix, node_attr, node_label, copy=False,
                          metadata={"node_class_name": ['days', 'mature', 'views', 'partner', 'new_id']})

        cache = dict(graph=graph)
        return cache

    @property
    def download_dir(self):
        return osp.join(self.root, 'musae', self.name)

    @property
    def raw_filenames(self):
        if self.attenuated:
            return ["attenuated_edges.csv", "features.json", "target.csv"]
        else:
            return ["edges.csv", "features.json", "target.csv"]

    @property
    def raw_paths(self):
        return [
            osp.join(self.download_dir, raw_filename)
            for raw_filename in self.raw_filenames
        ]


def encode(data, name):
    transformer = LabelEncoder()
    node_label = transformer.fit_transform(data[name])
    node_class_name = transformer.classes_
    return node_label, node_class_name
