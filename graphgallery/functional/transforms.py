from typing import Any

__all__ = ['Transform', 'NullTransform']


class Transform:

    def __init__(self):
        super().__init__()

    def __call__(self) -> Any:
        raise NotImplementedError


class NullTransform(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, inputs: Any) -> Any:
        return inputs

    def __repr__(self) -> str:
        return "NullTransform()"
