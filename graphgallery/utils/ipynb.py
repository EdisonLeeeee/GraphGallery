from IPython import get_ipython
from IPython.display import display


def is_ipynb():
    return type(get_ipython()).__module__.startswith('ipykernel.')
