import functools
import inspect

import graphgallery as gg
from typing import Callable, Any, List
from .functions import get_length, repeat

__all__ = ['multiple', 'equal', "wrapper"]


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


_BASE_VARS = ['hids', 'acts']


class equal:
    """
    A decorator class which makes the values of the 'variables' 
    equal in 'max-length'. 'Variables' consist of 'hids', 
    'acts' and other custom ones in `include`.

    """

    def __init__(self,
                 *,
                 include: List[str] = [],
                 exclude: List[str] = [],
                 length_as: str = 'hids'):
        """
        Parameters
        ----------
        include : list, optional
            the custom variable names except for 
            'hids', 'acts', by default []
        exclude : list, optional
            the excluded variable names, by default []
        length_as : str, optional
            the variable name whose length is used for all variables,
            by default 'hids'
        """
        _vars = list(include) + self.base_vars()
        _vars = list(set(_vars) - set(list(exclude)))
        assert length_as in _vars
        self._vars = _vars
        self.length_as = length_as

    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def _do_decoracte(*args, **kwargs) -> Any:
            ArgSpec = inspect.getfullargspec(func)

            if not ArgSpec.defaults or len(
                    ArgSpec.args) != len(ArgSpec.defaults) + 1:
                raise Exception(
                    f"'{func.__name__}' function must be defined with default parameters."
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

        return _do_decoracte

    @staticmethod
    def base_vars() -> List[str]:
        return _BASE_VARS


def wrapper(func: Callable) -> Callable:

    @functools.wraps(func)
    def decoracte(*args, **kwargs) -> Any:
        ArgSpec = inspect.getfullargspec(func)

        if not ArgSpec.defaults or len(
                ArgSpec.args) != len(ArgSpec.defaults) + 1:
            raise Exception(
                f"'{func.__name__}' function must be defined with default parameters."
            )
        values = list(args)
        for i in range(len(values), len(ArgSpec.args[1:])):
            values.append(ArgSpec.defaults[i])

        paras = dict(zip(ArgSpec.args[1:], values))
        paras.update(kwargs)

        accepted_vars = list(paras.get("include", [])) + ['hids', 'acts']
        accepted_vars = list(set(accepted_vars) - set(list(paras.get("exclude", []))))
        length_as = paras.get("length_as", "hids")
        assert length_as in accepted_vars

        repeated = get_length(paras.get(length_as, 0))
        for var in accepted_vars:
            if var in paras:
                val = paras[var]
                paras[var] = repeat(val, repeated)
        return func(**paras), paras

    return decoracte
