try:
    import texttable
except ImportError:
    texttable = None


def print_table(paras):
    assert texttable, "Please install `texttable` package!"
    if not isinstance(paras, dict):
        raise TypeError("The input should be the instance of `dict`.")
    t = texttable.Texttable()
    paras = paras.copy()
    name = paras.pop('name', None)
    sorted_keys = sorted(paras.keys())
    items = [(key, str(paras[key])) for key in sorted_keys]
    t.add_rows([['Parameters', 'Value'], ['Name', name], *list(items)])
    print(t.draw())
