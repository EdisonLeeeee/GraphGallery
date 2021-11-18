import graphgallery as gg
from graphgallery import functional as gf


gg.set_memory_growth()

from graphgallery.datasets import Planetoid, NPZDataset
data = NPZDataset('cora', root="~/GraphData/datasets/", verbose=False, transform='standardize')

graph = data.graph
splits = data.split_nodes(random_state=15)

for backend in ['th', 'dgl', 'pyg', 'tf']:
    gg.set_backend(backend)
    for device in ['cpu', 'cuda', 'gpu']:
        for name, m in gg.gallery.nodeclas.models():
            if name in ['LGCN', 'GraphMLP', 'PDN', 'ClusterGCN']:
                continue
            print(backend, device, name)
            trainer = m(device=device)
            trainer.setup_graph(graph, feat_transform=None)
            trainer.build()
            trainer.fit(splits.train_nodes, splits.val_nodes, verbose=0, epochs=2)
