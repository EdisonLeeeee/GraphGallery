import time
import sys
import numpy as np

import graphgallery as gg


class Progbar:
    """Displays a progress bar.

    Arguments:
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        interval: Minimum visual progress update interval (in seconds).
        unit_name: Display name for step counts (usually "step" or "sample").
    """

    def __init__(self,
                 target,
                 width=30,
                 verbose=1,
                 interval=0.05,
                 unit_name='step'):

        assert gg.is_intscalar(target), target
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        self.unit_name = unit_name

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules or
                                 'posix' in sys.modules or
                                 'PYCHARM_HOSTED' in os.environ)
        self._total_width = 0
        self._seen_so_far = 0
        self._start = time.perf_counter()
        self._last_update = 0

    def update(self, current, values=None, finalize=None):
        """Updates the progress bar.

        Arguments:
            current: Index of current step.
            values: List of tuples: `(name, value_for_last_step)` or string messages.
            finalize: Whether this is the last update for the progress bar. If
              `None`, defaults to `current >= self.target`.
        """
        if finalize is None:
            if self.target is None:
                finalize = False
            else:
                finalize = current >= self.target
        values = values or []

        if isinstance(values, str):
            message = ' - ' + values
        else:
            message = ''
            for k, v in values:
                message += ' - %s:' % k
                if v is None:
                    message += ' None'
                elif v > 1e-3:
                    message += ' %.3f' % v
                else:
                    message += ' %.3e' % v

        self._seen_so_far = current

        now = time.perf_counter()
        delta = now - self._start

        if delta >= 1:
            delta = ' %.2fs' % delta
        elif delta >= 1e-3:
            delta = ' %.2fms' % (delta * 1e3)
        else:
            delta = ' %.2fus' % (delta * 1e6)
        info = ' - Total:%s' % delta

        if self.verbose == 1:
            if now - self._last_update < self.interval and not finalize:
                return
            info += ' -'
            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.log10(self.target)) + 1
                bar = ('%' + str(numdigits) + 'd/%d [') % (current, self.target)
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0

            if self.target is None or finalize:
                if time_per_unit >= 1 or time_per_unit == 0:
                    info += ' %ds/%s' % (time_per_unit, self.unit_name)
                elif time_per_unit >= 1e-3:
                    info += ' %dms/%s' % (time_per_unit * 1e3, self.unit_name)
                else:
                    info += ' %dus/%s' % (time_per_unit * 1e6, self.unit_name)
            else:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                   (eta % 3600) // 60, eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format

            info += message

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if finalize:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if finalize:
                numdigits = int(np.log10(self.target)) + 1
                count = ('%' + str(numdigits) + 'd/%d') % (current, self.target)
                info = count + info
                info += message
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)
