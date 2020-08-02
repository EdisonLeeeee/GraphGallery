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


def set_equal_in_length(*inputs, max_length=None):
    """ Check if the inputs are lists or tuples,
        and convert them into lists with the same lengths (max lengths
        of the inputs). The shorter ones will repeted.

        Arguments:
        ----------
        inputs: list of scalar, list, tuple.
        max_length: The maximum length of the input list.

        Returns:
        ----------
        outputs: list of lists with the same lengths.        

    """
    lengths = []
    outputs = []
    for para in inputs:
        length = get_length(para)
        outputs.append(para)
        lengths.append(length)

    max_length = max_length or max(lengths)
    for idx, length in enumerate(lengths):
        outputs[idx] = repeat(outputs[idx], max_length)
    return outputs


def get_length(arr):
    if isinstance(arr, (list, tuple)):
        length = len(arr)
    else:
        length = 1
    return length
