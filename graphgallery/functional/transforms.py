from abc import ABC
from typing import Any

__all__ = ['Transform', 'NullTransform']

class Transform(ABC):

    def __init__(self):
        """
        Initialize the state

        Args:
            self: (todo): write your description
        """
        super().__init__()

    def __call__(self)  -> Any :
        """
        Call the call.

        Args:
            self: (todo): write your description
        """
        raise NotImplementedError


class NullTransform(Transform):
    def __init__(self, *args, **kwargs):
        """
        Initialize the init.

        Args:
            self: (todo): write your description
        """
        super().__init__()

    def __call__(self, inputs: Any) -> Any:
        """
        Call the given inputs.

        Args:
            self: (todo): write your description
            inputs: (dict): write your description
        """
        return inputs

    def __repr__(self) -> str:
        """
        Return a repr representation of - repr representation.

        Args:
            self: (todo): write your description
        """
        return "NullTransform()"
