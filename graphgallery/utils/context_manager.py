
from typing import Any
from contextlib import contextmanager


@contextmanager
def nullcontext(enter_result=None):
    """Null context manager.
    Nothing is done when it's called

    Parameters
    ----------
    enter_result : Any, optional
        any objects, by default None

    Returns
    -------
    return the inputs

    Yields
    -------
    Iterator[Any]
        yields the inputs
    """
    yield enter_result
