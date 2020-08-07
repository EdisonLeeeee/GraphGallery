"""Inspired by Keras backend config API. https://tensorflow.google.com """


from tensorflow.keras import backend as K



floatx = K.floatx
set_floatx = K.set_floatx
epsilon = K.epsilon
set_epsilon = K.set_epsilon

_INTX = 'int32'

def intx():
    """Returns the default integer type, as a string.

    E.g. `'int16'`, `'int32'`, `'int64'`.

    Returns:
      String, the current default integer type.

    Example:
    >>> graphgallery.config.intx()
    'int64'
    """       
    return _INTX


def set_intx(dtype):
    """Sets the default integer type.

    Arguments:
      value: String; `'int16'`, `'int32'`, or `'int64'`.

    Example:
    >>> graphgallery.config.intx()
    'int64'
    >>> graphgallery.config.set_intx('int32')
    >>> graphgallery.config.intx()
    'int32'

    Raises:
      ValueError: In case of invalid value.
    """ 
    
    if dtype not in {'int16', 'int32', 'int64'}:
        raise ValueError('Unknown floatx type: ' + str(dtype))
    global _INTX
    _INTX = dtype

