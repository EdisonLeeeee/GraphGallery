import torch.nn.functional as F


def get_activation(activation_string):
    if activation_string is None:
        return lambda x : x
    
    if not isinstance(activation_string, str):
        raise ValueError(f"'activation_string' is expected a `string`, "
                         "but got {type(activation_string)}.")
        
    activation_fn = getattr(F, activation_string.lower(), 'None')
    
    if activation_fn:
        return activation_fn
    else:
        raise KeyError(
            f"function {activation_string} not found in torch.nn.functional."
        )