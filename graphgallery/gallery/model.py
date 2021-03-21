import graphgallery as gg
from graphgallery import functional as gf


class Model:
    def __init__(self, *, device="cpu", seed=None, name=None, **kwargs):
        """
        Parameters:
        ----------
        device: string. optional
            The device where the model running on.
        seed: interger scalar. optional
            Used to create a reproducible sequence of tensors
            across multiple calls.
        name: string. optional
            Specified name for the model. (default: :str: `class name`)
        kwargs: other custom keyword arguments. 
        """
        # if graph is not None and not isinstance(graph, gg.data.BaseGraph):
        #     raise ValueError(f"Unrecognized graph: {graph}.")

        # It currently takes no keyword arguments
        gg.utils.raise_error.raise_if_kwargs(kwargs)

        _backend = gg.backend()

        if seed:
            gf.random_seed(seed, _backend)

        self.seed = seed
        self.name = name or self.__class__.__name__
        self.device = gf.device(device, _backend)
        self.data_device = self.device
        self.backend = _backend

        # data types, default: `float32`,`int32` and `bool`
        self.floatx = gg.floatx()
        self.intx = gg.intx()
        self.boolx = gg.boolx()
        self._cache = gf.BunchDict()
        self.transform = gf.BunchDict()

        self.cfg = None
        self._model = None
        self._graph = None
        self.setup_cfg()

    def setup_cfg(self):
        pass

    @property
    def graph(self):
        graph = self._graph
        if graph is None:
            raise KeyError("You haven't pass any graph instance.")
        return graph

    @graph.setter
    def graph(self, graph):
        assert graph is None or isinstance(graph, gg.data.BaseGraph)
        if graph is not None:
            self._graph = graph.copy()

    @property
    def cache(self):
        return self._cache

    def register_cache(self, **kwargs):
        self._cache.update(kwargs)

    def __repr__(self):
        return f"{self.name}(device={self.device}, backend={self.backend})"

    __str__ = __repr__
