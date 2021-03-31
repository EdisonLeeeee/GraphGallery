import six
from tabulate import tabulate

__all__ = ["dict_to_string", "merge_as_list",
"ask_to_proceed_with_overwrite", "create_table"]


def create_table(small_dict):
    """
    Create a small table using the keys of small_dict as headers. This is only
    suitable for small dictionaries.
    Args:
        small_dict (dict): a result dictionary of only a few items.
    Returns:
        str: the table as a string.
    """
    keys, values = tuple(zip(*small_dict.items()))
    table = tabulate(
        [values],
        headers=keys,
        tablefmt="pipe",
        floatfmt=".3f",
        stralign="center",
        numalign="center",
    )
    return table


def dict_to_string(d, fmt="%.4f"):
    s = ""
    for k, v in d.items():
        fmt_string = "%s: " + fmt + " - "
        s += fmt_string % (k, v)
    if d:
        s = s[:-2]
    return s

def merge_as_list(*args):
    out = []
    for x in args:
        if x is not None:
            if isinstance(x, (list, tuple)):
                out += x
            else:
                out += [x]
    return out


def ask_to_proceed_with_overwrite(filepath):
    """Produces a prompt asking about overwriting a file.

    Parameters:
      filepath: the path to the file to be overwritten.

    Returns:
      True if we can proceed with overwrite, False otherwise.
    """
    overwrite = six.moves.input('[WARNING] %s already exists - overwrite? '
                                '[y/n]' % (filepath)).strip().lower()
    while overwrite not in ('y', 'n'):
        overwrite = six.moves.input('Enter "y" (overwrite) or "n" '
                                    '(cancel).').strip().lower()
    if overwrite == 'n':
        return False
    print('[TIP] Next time specify overwrite=True!')
    return True
