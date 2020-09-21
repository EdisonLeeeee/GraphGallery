import itertools


from numbers import Number


def repeat(src, length):
    if src is None:
        return [None for _ in range(length)]
    if isinstance(src, (Number, str)):
        return list(itertools.repeat(src, length))
    if (len(src) > length):
        return src[:length]
    if (len(src) < length):
        return list(src) + list(itertools.repeat(src[-1], length - len(src)))
    return src
