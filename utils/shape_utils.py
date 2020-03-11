import itertools
from numbers import Number


def repeat(src, length):
    if src is None:
        return None
    if isinstance(src, Number):
        return list(itertools.repeat(src, length))
    if (len(src) > length):
        return src[:length]
    if (len(src) < length):
        return src + list(itertools.repeat(src[-1], length - len(src)))
    return src

