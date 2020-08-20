from graphgallery.data import ZippedGraphDataset
from graphgallery.data.utils import train_val_test_split_tabular
from graphgallery.utils.graph import largest_connected_components


class SimpleGraphDataset(ZippedGraphDataset):

    supported_datasets = ('citeseer', 'cora', 'cora_ml', 'polblogs', 'pubmed', 'reddit')

    def __init__(self, name, train_size=0.1, val_size=0.1, test_size=0.8,
                 root=None, url=None, seed=None, largest_cc=False):
        name = name.lower()

        if not name in self.supported_datasets:
            raise ValueError(f"Currently only support for these datasets {self.supported_datasets}.")

        super().__init__(name, root, url)

        adj, x, labels = self.load()

        if largest_cc:
            '''select the largest connected components (LCC)'''
            idx = largest_connected_components(adj)
            adj = adj[idx].tocsc()[:, idx]
            x = x[idx]
            labels = labels[idx]

        idx_train, idx_val, idx_test = train_val_test_split_tabular(adj.shape[0],
                                                                    train_size=train_size,
                                                                    val_size=val_size,
                                                                    test_size=test_size,
                                                                    stratify=labels,
                                                                    random_state=seed)

        self.adj, self.x, self.labels = adj, x, labels
        self.idx_train, self.idx_val, self.idx_test = idx_train, idx_val, idx_test
