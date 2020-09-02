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


class SetEqual:
    """
    A decorator class which makes the values of the variables 
    equal in max-length. variables consist of 'hiddens', 'activations', 
    'dropouts', 'l2_norms' and other customed ones in `var_names`.
    
    """
    base_var_names = ['hiddens', 'activations', 'dropouts', 'l2_norms']
    
    def __init__(self, var_names=[]):
        """
        var_names: string, a list or tuple of string.
            the customed variable name except for 'hiddens', 'activations', 
            'dropouts', 'l2_norms'. 
        """
        self.var_names = list(var_names) + self.base_var_names
        
    def __call__(self, func):
        
        @functools.wraps(func)
        def set_equal_in_length(*args, **kwargs):
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
        return set_equal_in_length


def get_length(arr):
    if isinstance(arr, (list, tuple)):
        length = len(arr)
    else:
        length = 1
    return length
