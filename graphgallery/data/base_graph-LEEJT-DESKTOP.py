from abc import ABC


class base_graph(ABC):

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

    @property
    def A(self):
        """alias of adj_matrix"""
        return self.adj_matrix

    @property
    def X(self):
        """alias of attr_matrix"""
        return self.attr_matrix

    @property
    def Y(self):
        """alias of labels"""
        return self.labels

    def copy(self, deepcopy=False):
        if deepcopy:
            new_graph = type(self)(*self.unpack(), node_names=self.node_names,
                                   attr_names=self.attr_names, class_names=self.class_names,
                                   metadata=self.metadata, copy=True)
        else:
            new_graph = type(self).__new__(type(self))
            new_graph.__dict__ = self.__dict__
            print(self.__dict__)
        return new_graph
