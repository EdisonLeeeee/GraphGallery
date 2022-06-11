from collections import OrderedDict

from tabulate import tabulate


class BunchDict(OrderedDict):
    """Container object for datasets
    Dictionary-like object that exposes its keys as attributes
    and remembers insertion order.

    Examples
    --------
    >>> b = BunchDict(a=1, b=2)
    >>> b
    Objects in BunchDict:
    ╒═════════╤═══════════╕
    │ Names   │   Objects │
    ╞═════════╪═══════════╡
    │ a       │         1 │
    ├─────────┼───────────┤
    │ b       │         2 │
    ╘═════════╧═══════════╛
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

    >>> # Converting objects in BunchDict to `torch.Tensor` if possible.
    >>> b = BunchDict(a=[1,2,3])
    >>> b.to_tensor()
    Objects in BunchDict:
    ╒═════════╤═══════════════════════════════╕
    │ Names   │ Objects                       │
    ╞═════════╪═══════════════════════════════╡
    │ a       │ Tensor, shape=torch.Size([3]) │
    │         │ tensor([1, 2, 3])             │
    ╘═════════╧═══════════════════════════════╛
    >>> b.a
    tensor([1, 2, 3])
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

    def to_tensor(self, device: str = 'cpu', dtype=None) -> "BunchDict":
        """Convert objects in BunchDict to :class:`torch.Tensor`

        Parameters
        ----------
        device : str, optional
            device of the converted tensors, by default 'cpu'
        dtype : _type_, optional
            data types of the converted tensors, by default None

        Returns
        -------
        the converted BunchDict
        """
        import torch
        device = torch.device(device)
        for k, v in self.items():
            try:
                self[k] = torch.as_tensor(v, dtype=dtype, device=device)
            except RuntimeError:
                pass
        return self

    def __repr__(self) -> str:
        table_headers = ["Names", "Objects"]
        items = tuple(map(prettify, self.items()))
        table = tabulate(
            items, headers=table_headers, tablefmt="fancy_grid"
        )
        return "Objects in BunchDict:\n" + table

    __str__ = __repr__


def prettify(item):
    key, val = item
    if val is None:
        return key, 'None'
    if hasattr(val, "shape"):
        if len(val.shape) == 0 and hasattr(val, "item"):
            val = f"{val.__class__.__name__}, {val.item()}"
        else:
            val = f"{val.__class__.__name__}, shape={val.shape}\n{val}"
    else:
        if isinstance(val, str):
            val = f"{val}"
        else:
            try:
                val = f"{type(val).__name__}, len={len(val)}"
            except TypeError:
                pass
    return key, val
