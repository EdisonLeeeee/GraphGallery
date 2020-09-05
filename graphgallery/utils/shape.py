import itertools
import inspect
import functools

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


_BASE_VARS = ['hiddens', 'activations', 'dropouts', 'l2_norms']


class EqualVarLength:
    """
    A decorator class which makes the values of the variables 
    equal in max-length. variables consist of 'hiddens', 'activations', 
    'dropouts', 'l2_norms' and other customed ones in `var_names`.

    """

    def __init__(self, include=[], exclude=[]):
        """
        include: string, a list or tuple of string, optional.
            the customed variable name except for 'hiddens', 'activations', 
            'dropouts', 'l2_norms'. 
        exclude: string, a list or tuple of string, optional.
            the exclued variable name.
        """
        self.var_names = list(include) + self.base_vars()
        self.var_names = list(set(self.var_names)-set(exclude))

    def __call__(self, func):

        @functools.wraps(func)
        def decorator(*args, **kwargs):
            ArgSpec = inspect.getfullargspec(func)

            if not ArgSpec.defaults or len(ArgSpec.args) != len(ArgSpec.defaults) + 1:
                raise Exception(f"The '{func.__name__}' method must be defined with all default parameters.")

            model, *values = args
            for i in range(len(values), len(ArgSpec.args[1:])):
                values.append(ArgSpec.defaults[i])

            paras = dict(zip(ArgSpec.args[1:], values))
            paras.update(kwargs)

            max_length = 0
            for var in self.var_names:
                val = paras.get(var, None)
                if val is not None:
                    max_length = max(get_length(val), max_length)

            for var in self.var_names:
                val = paras.get(var, None)
                if val is not None:
                    paras[var] = repeat(val, max_length)

            return func(model, **paras)
        return decorator

    @staticmethod
    def base_vars():
        return _BASE_VARS


def get_length(arr):
    if isinstance(arr, (list, tuple)):
        length = len(arr)
    else:
        length = 1
    return length
