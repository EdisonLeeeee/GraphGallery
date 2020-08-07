from contextlib import contextmanager

@contextmanager
def nullcontext(enter_result=None):
    yield enter_result