"""Inspired by Keras backend config API. https://tensorflow.google.com """


from tensorflow.keras import backend as K


epsilon = K.epsilon
set_epsilon = K.set_epsilon

# The type of integer to use throughout a session.
_INTX = 'int32'

# The type of float to use throughout a session.
_FLOATX = 'float32'


def floatx():
    """Returns the default float type, as a string.

    E.g. `'float16'`, `'float32'`, `'float64'`.

    Returns:
      String, the current default float type.

    Example:
    >>> graphgallery.floatx()
    'float32'
    """
    return _FLOATX


def set_floatx(dtype):
    """Sets the default float type.

    Parameters:
      value: String; `'float16'`, `'float32'`, or `'float64'`.

    Example:
    >>> graphgallery.floatx()
    'float32'
    >>> graphgallery.set_floatx('float64')
    >>> graphgallery.floatx()
    'float64'

    Raises:
      ValueError: In case of invalid value.
    """

    if dtype not in {'float16', 'float32', 'float64'}:
        raise ValueError('Unknown floatx type: ' + str(dtype))
    global _FLOATX
    _FLOATX = str(dtype)


def intx():
    """Returns the default integer type, as a string.

    E.g. `'int16'`, `'int32'`, `'int64'`.

    Returns:
      String, the current default integer type.

    Example:
    >>> graphgallery.intx()
    'int32'
    """
    return _INTX


def set_intx(dtype):
    """Sets the default integer type.

    Parameters:
      value: String; `'int16'`, `'int32'`, or `'int64'`.

    Example:
    >>> graphgallery.intx()
    'int32'
    >>> graphgallery.set_intx('int64')
    >>> graphgallery.intx()
    'int64'

    Raises:
      ValueError: In case of invalid value.
    """

    if dtype not in {'int16', 'int32', 'int64'}:
        raise ValueError('Unknown floatx type: ' + str(dtype))
    global _INTX
    _INTX = str(dtype)
