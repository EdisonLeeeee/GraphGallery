import functools
import inspect

import graphgallery as gg
from typing import Callable, Any, List
from .functions import get_length, repeat

__all__ = ['multiple', 'equal']


def cal_outpus(func: Callable,
               args: list,
               kwargs: dict,
               type_check: bool = True):

    if gg.is_multiobjects(args):
        if type_check:
            assert_same_type(*args)
        return tuple(
            cal_outpus(func, arg, kwargs, type_check=type_check)
            for arg in args)

    return func(args, **kwargs)


class multiple:

    wrapper_doc = """This method is decorated by 
    'graphgallery.functional.multiple',
    which takes multiple inputs and yields multiple outputs.
    """

    def __init__(self, *, type_check: bool = True):
        self.type_check = type_check

    def __call__(self, func: Callable) -> Callable:
        doc = func.__doc__ if func.__doc__ else ""
        func.__doc__ = doc + self.wrapper_doc

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if len(args) == 1 and gg.is_multiobjects(args[0]):
                args, = args

            outputs = cal_outpus(func,
                                 args,
                                 kwargs,
                                 type_check=self.type_check)
            if outputs is not None and gg.is_multiobjects(outputs) and len(
                    outputs) == 1:
                outputs, = outputs
            return outputs

        return wrapper


def assert_same_type(*inputs) -> bool:
    """ Assert the types of inputs are the same"""
    first, *others = inputs
    # single input
    if not others:
        return True

    _class = type(first)
    for ix, obj in enumerate(others):
        if not isinstance(obj, _class):
            raise TypeError(f"Input types don't agree. "
                            f"Type of the first input: {type(first)}, "
                            f"but {ix+1}-th input: {type(obj)}")

    return True


_BASE_VARS = ['hiddens', 'activations']


class equal:
    """
    A decorator class which makes the values of the 'variables' 
    equal in 'max-length'. 'Variables' consist of 'hiddens', 
    'activations' and other custom ones in `include`.

    """

    def __init__(self,
                 *,
                 include: List[str] = [],
                 exclude: List[str] = [],
                 length_as: str = 'hiddens'):
        """
        Parameters
        ----------
        include : list, optional
            the custom variable names except for 
            'hiddens', 'activations', by default []
        exclude : list, optional
            the excluded variable names, by default []
        length_as : str, optional
            the variable name whose length is used for all variables,
            by default 'hiddens'
        """
        _vars = list(include) + self.base_vars()
        _vars = list(set(_vars) - set(list(exclude)))
        assert length_as in _vars
        self._vars = _vars
        self.length_as = length_as

    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            ArgSpec = inspect.getfullargspec(func)

            if not ArgSpec.defaults or len(
                    ArgSpec.args) != len(ArgSpec.defaults) + 1:
                raise Exception(
                    f"The '{func.__name__}' method must be defined with all default parameters."
                )

            model, *values = args
            for i in range(len(values), len(ArgSpec.args[1:])):
                values.append(ArgSpec.defaults[i])

            paras = dict(zip(ArgSpec.args[1:], values))
            paras.update(kwargs)

            repeated = get_length(paras.get(self.length_as, 0))
            for var in self._vars:
                # use `NAN` instead of `None` to avoid `None` exists
                val = paras.get(var, "NAN")
                if val != "NAN":
                    paras[var] = repeat(val, repeated)

            return func(model, **paras)

        return wrapper

    @staticmethod
    def base_vars() -> List[str]:
        return _BASE_VARS
