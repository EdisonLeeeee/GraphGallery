from typing import Any

__all__ = ['Transform', 'NullTransform']


class Transform:

    def __init__(self):
        super().__init__()

    def __call__(self) -> Any:
        raise NotImplementedError

    def extra_repr(self):
        return ""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"


class NullTransform(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, inputs: Any) -> Any:
        return inputs
