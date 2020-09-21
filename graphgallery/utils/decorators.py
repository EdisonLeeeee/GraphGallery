import functools
import inspect

from graphgallery.utils.type_check import is_list_like, is_scalar_like
from graphgallery.utils.shape import repeat


def cal_outpus(func, args, kwargs, type_check=True):

    if is_list_like(args) and not is_scalar_like(args[0]):
        if type_check:
            assert_same_type(*args)
        return tuple(cal_outpus(func, arg, kwargs, type_check=type_check) for arg in args)

    return func(args, **kwargs)


class MultiInputs:

    def __init__(self, *, type_check=True):
        self.type_check = type_check

    def __call__(self, func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if len(args) == 1 and is_list_like(args[0]):
                args, = args

            outputs = cal_outpus(func, args, kwargs,
                                 type_check=self.type_check)
            if outputs is not None and is_list_like(outputs) and len(outputs) == 1:
                outputs, = outputs
            return outputs

        return wrapper


def assert_same_type(*inputs):
    first, *others = inputs
    # only one inputs
    if not others:
        return True

    _class = type(first)
    for ix, obj in enumerate(others):
        if not isinstance(obj, _class):
            raise TypeError(f"Input types don't agree. "
                            f"Type of the first input: {type(first)}, "
                            f"{ix+1}th input: {type(obj)}")

    return True


_BASE_VARS = ['hiddens', 'activations', 'dropouts', 'l2_norms']


class EqualVarLength:
    """
    A decorator class which makes the values of the variables 
    equal in max-length. variables consist of 'hiddens', 'activations', 
    'dropouts', 'l2_norms' and other customed ones in `include`.

    """

    def __init__(self, *, include=[], exclude=[]):
        """
        include: string, a list or tuple of string, optional.
            the customed variable name except for 'hiddens', 'activations', 
            'dropouts', 'l2_norms'. 
        exclude: string, a list or tuple of string, optional.
            the exclued variable name.
        """
        self.var_names = list(include) + self.base_vars()
        self.var_names = list(set(self.var_names) - set(list(exclude)))

    def __call__(self, func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            ArgSpec = inspect.getfullargspec(func)

            if not ArgSpec.defaults or len(ArgSpec.args) != len(ArgSpec.defaults) + 1:
                raise Exception(
                    f"The '{func.__name__}' method must be defined with all default parameters.")

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
        return wrapper

    @staticmethod
    def base_vars():
        return _BASE_VARS


def get_length(arr):
    if isinstance(arr, (list, tuple)):
        length = len(arr)
    else:
        length = 1
    return length
