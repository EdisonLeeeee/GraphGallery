from typing import Dict, Any


def raise_if_kwargs(kwargs: Dict[str, Any]) -> None:
    """
    Check if kwargs.

    Args:
    """
    if kwargs:
        raise TypeError(
            f"Got an unexpected keyword argument '{next(iter(kwargs.keys()))}'"
        )
        