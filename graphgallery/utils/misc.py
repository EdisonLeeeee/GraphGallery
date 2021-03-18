import six
__all__ = ["dict_to_string", "merge_as_list", "ask_to_proceed_with_overwrite"]

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
