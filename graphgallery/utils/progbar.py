# The following codes are mainly modified from tensorflow and some changes are made.
# You may refer to tensorflow for more details:
#
#     https://github.com/tensorflow/tensorflow
#
# Copyright The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import time
import sys
import os
import numpy as np
from numbers import Number


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

    def update(self, current, msg=None, finalize=None):
        """Updates the progress bar.

        Arguments:
            current: Index of current step.
            msg: List of tuples: `(name, value_for_last_step)` or string messages.
            finalize: Whether this is the last update for the progress bar. If
              `None`, defaults to `current >= self.target`.
        """
        if not self.verbose:
            return

        if finalize is None:
            if self.target is None:
                finalize = False
            else:
                finalize = current >= self.target
        msg = msg or {}

        if isinstance(msg, str):
            message = ' - ' + msg
        elif isinstance(msg, (dict, list, tuple)):
            message = ''
            if isinstance(msg, dict):
                msg = msg.items()
            else:
                assert len(msg[0]) == 2
            for k, v in msg:
                message += ' - %s:' % k
                if v is None:
                    message += ' None'
                elif isinstance(v, str):
                    message += ' ' + v
                else:
                    message += ' ' + self.format_num(v)
        else:
            raise ValueError(msg)

        message = message.strip()

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
            sys.stdout.write(f'{bar}{info}')
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

    def add(self, n, msg=None):
        self.update(self._seen_so_far + n, msg)

    @staticmethod
    def format_num(n):
        """
        Intelligent scientific notation (.3g).

        Parameters
        ----------
        n  : int or float or Numeric
            A Number.

        Returns
        -------
        out  : str
            Formatted number.
        """
        assert isinstance(n, Number), f'{n} is not a Number.'
        f = '{0:.3g}'.format(n).replace('+0', '+').replace('-0', '-')
        n = str(n)
        return f if len(f) < len(n) else n
