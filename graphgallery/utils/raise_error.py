
def raise_if_kwargs(kwargs):
    if kwargs:
        raise TypeError(
            f"Got an unexpected keyword argument '{next(iter(kwargs.keys()))}'"
        )
