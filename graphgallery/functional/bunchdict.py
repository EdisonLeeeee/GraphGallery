from collections import OrderedDict

class BunchDict(OrderedDict):
    """Container object for datasets
    Dictionary-like object that exposes its keys as attributes
    and remembers insertion order.
    >>> b = Bunch(a=1, b=2)
    >>> b['b']
    2
    >>> b.b
    2
    >>> b.a = 3
    >>> b['a']
    3
    >>> b.c = 6
    >>> b['c']
    6
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def show(self):
        # TODO: improve it using texttable
        return f"{self.__class__.__name__}({tuple(self.keys())})"