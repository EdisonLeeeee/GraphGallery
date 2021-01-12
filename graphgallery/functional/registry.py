# inspired from fvcore:
# https://github.com/facebookresearch/fvcore
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, Iterable, Iterator, Optional, Tuple
from tabulate import tabulate


class Registry(Iterable[Tuple[str, object]]):
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.

    To create a registry (e.g. a backbone registry):

    .. code-block:: python

        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:

    .. code-block:: python

        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...

    Or:

    .. code-block:: python

        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name: str) -> None:
        """
        Args:
            name (str): the name of this registry
        """
        self._name: str = name
        self._obj_map: Dict[str, object] = {}

    def _do_register(self, name: str, obj: object, freeze: bool = True) -> None:
        if freeze:
            assert (
                name not in self._obj_map
            ), "An object named '{}' was already registered in '{}' registry!".format(
                name, self._name
            )
        self._obj_map[name] = obj

    def register(self, obj: object = None, *, name: str = None, freeze: bool = False) -> Optional[object]:
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class: object) -> object:
                nonlocal name, freeze
                if name is None:
                    try:
                        name = func_or_class.__name__  # pyre-ignore
                    except AttributeError:
                        name = str(obj)
                self._do_register(name, func_or_class, freeze=freeze)
                return func_or_class

            return deco

        # used as a function call
        if name is None:
            try:
                name = obj.__name__  # pyre-ignore
            except AttributeError:
                name = str(obj)
        self._do_register(name, obj, freeze=freeze)

    def get(self, name: str) -> object:
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                "No object named '{}' found in '{}' registry!".format(name, self._name)
            )
        return ret
    # __getattr__ = get

    def items(self):
        return tuple(self._obj_map.items())

    def names(self):
        return tuple(self._obj_map.keys())

    def objects(self):
        return tuple(self._obj_map.values())

    def __dir__(self):
        return self.keys()

    def __add__(self, register):
        assert isinstance(register, Registry)
        a = self._obj_map
        b = register._obj_map
        c = dict(**a, **b)
        merged = type(self)(self._name)
        merged._obj_map = c
        return merged

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def __getattr__(self, name: str) -> object:
        return self.get(name)

    def __repr__(self) -> str:
        table_headers = ["Names", "Objects"]
        table = tabulate(
            self._obj_map.items(), headers=table_headers, tablefmt="fancy_grid"
        )
        return "Registry of {}:\n".format(self._name) + table

    def __len__(self):
        return len(self._obj_map)

    def __iter__(self) -> Iterator[Tuple[str, object]]:
        return iter(self._obj_map.items())

    # pyre-fixme[4]: Attribute must be annotated.
    __str__ = __repr__
