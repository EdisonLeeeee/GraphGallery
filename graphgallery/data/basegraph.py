from abc import ABC


class BaseGraph(ABC):

    def __init__(self,):
        ...

    @property
    def n_nodes(self):
        ...

    @property
    def n_edges(self):
        ...

    @property
    def n_graphs(self):
        ...

    @property
    def n_attrs(self):
        ...

    @property
    def n_classes(self):
        pass
