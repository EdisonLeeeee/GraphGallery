"""
Copyright (C) scikit-network team
https://github.com/sknetwork-team/scikit-network
"""
import contextlib
import signal


class TimeOut(contextlib.ContextDecorator):
    """
    Timeout context manager/decorator.

    Adapted from https://gist.github.com/TySkby/143190ad1b88c6115597c45f996b030c on 12/10/2020.

    Examples
    --------
    >>> from time import sleep
    >>> try:
    ...     with TimeOut(1):
    ...         sleep(10)
    ... except TimeoutError:
    ...     print("Function timed out")
    Function timed out
    """

    def __init__(self, seconds: float):
        self.seconds = seconds

    def _timeout_handler(self, signum, frame):
        raise TimeoutError("Code timed out.")

    def __enter__(self):
        signal.signal(signal.SIGALRM, self._timeout_handler)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)
