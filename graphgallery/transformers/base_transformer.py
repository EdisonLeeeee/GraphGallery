from abc import ABC
from typing import Any


class Transformer(ABC):

    def __init__(self):
        ...

    def __call__(self):
        ...


class NullTransformer(Transformer):
    def __init__(self, *args, **kwargs):
        ...

    def __call__(self, inputs: Any):
        return inputs

    def __repr__(self):
        return "NullTransformer()"
