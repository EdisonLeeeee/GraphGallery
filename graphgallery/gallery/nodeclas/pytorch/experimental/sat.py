import scipy.sparse as sp

from graphgallery.data.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model


@PyTorch.register()
class SAT(Trainer):
    def data_step(self,
                  adj_transform="normalize_adj",
                  attr_transform=None,
                  k=35):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(attr_transform)(graph.attr_matrix)

        V, U = sp.linalg.eigsh(adj_matrix, k=k)

        adj_matrix = (U * V) @ U.T
        adj_matrix[adj_matrix < 0] = 0.
        adj_matrix = gf.get(adj_transform)(adj_matrix)

        X, A, U, V = gf.astensors(attr_matrix,
                                  adj_matrix,
                                  U,
                                  V,
                                  device=self.data_device)

        # ``A`` , ``X`` , U`` and ``V`` are cached for later use
        self.register_cache(X=X, A=A, U=U, V=V)

    def model_step(self,
                   hids=[32],
                   acts=['relu'],
                   K=5,
                   alpha=0.2,
                   eps_U=0.1,
                   eps_V=0.1,
                   lamb_U=0.5,
                   lamb_V=0.5,
                   dropout=0.5,
                   weight_decay=5e-4,
                   lr=0.01,
                   bias=True,
                   name="sat.SSGC"):

        model = get_model(name, self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      K=K,
                      alpha=alpha,
                      eps_U=eps_U,
                      eps_V=eps_V,
                      lamb_U=lamb_U,
                      lamb_V=lamb_V,
                      hids=hids,
                      acts=acts,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      bias=bias)

        return model

    def train_loader(self, index):

        labels = self.graph.label[index]
        sequence = FullBatchSequence(inputs=[self.cache.X, self.cache.U, self.cache.V],
                                     y=labels,
                                     out_index=index,
                                     device=self.data_device)
        return sequence

    def test_loader(self, index):

        labels = self.graph.label[index]
        sequence = FullBatchSequence(inputs=[self.cache.X, self.cache.A],
                                     y=labels,
                                     out_index=index,
                                     device=self.data_device)
        return sequence
