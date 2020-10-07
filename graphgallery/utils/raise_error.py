from typing import Dict, Any


def raise_if_kwargs(kwargs: Dict[str, Any]) -> None:
    if kwargs:
        raise TypeError(
            f"Got an unexpected keyword argument '{next(iter(kwargs.keys()))}'"
        )
        
def assert_kind(kind: Any) -> None:
    """assert if `kind` is `'T'` (tensorflow)
    or `'P'`(PyTorch)

    Parameters
    ----------
    kind : Any
        the kind of backend (tensorflow or pytorch)

    Raises
    ------
    ValueError
        kind not in {"T", "P"}
    """
    if not kind in {"T", "P"}:
        raise ValueError(f"`kind` should be 'T' or 'P', but got {kind}.")
