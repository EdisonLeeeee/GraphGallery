from ..bunchdict import BunchDict
from ..transform import GraphTransform
from ..transform import Transform, get


@Transform.register()
class DictTransform(GraphTransform):

    def __init__(self, **transforms):
        super().__init__()
        _transforms = BunchDict()
        for k, v in transforms.items():
            _transforms[k] = get(v)
        self.transforms = _transforms

    def __call__(self, graph):
        transforms = self.transforms
        names = transforms.keys()
        updates = {}
        for name, value in zip(names, graph(*names)):
            updates[name] = transforms[name](value)
        graph = graph.copy()
        graph.update(**updates)
        return graph

    def extra_repr(self):
        return f"Transforms={tuple(self.transforms.items())}"
