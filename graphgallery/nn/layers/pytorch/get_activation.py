import torch.nn.functional as F


def get_activation(activation):
    if callable(activation):
        return activation
    
    if activation is None:
        return lambda x : x
    
    if not isinstance(activation, str):
        raise ValueError(f"'activation' is expected a `string`, "
                         "but got {type(activation)}.")
        
    activation_fn = getattr(F, activation.lower(), 'None')
    
    if activation_fn:
        return activation_fn
    else:
        raise KeyError(
            f"function {activation} not found in torch.nn.functional."
        )
