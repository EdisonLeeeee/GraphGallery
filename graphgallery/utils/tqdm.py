# from IPython import get_ipython
from tqdm import tqdm as tqdm_base


def tqdm_clear(*args, **kwargs):
    getattr(tqdm_base, '_instances', {}).clear()


def tqdm(*args, **kwargs):

    #     get_ipython().events.register('post_execute', tqdm_clear)
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)

tqdm.__doc__ = tqdm_base.__doc__ + tqdm_base.__init__.__doc__