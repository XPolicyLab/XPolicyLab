from datetime import datetime
import io
import logging
import os
import pprint
import shlex
import string
import sys
import textwrap
import time
import traceback
from typing import Callable, Union

import numpy as np
from typing_extensions import Literal

from ..misc.functional_utils import meta_decorator
from ..misc.misc_utils import match_patterns


def to_readable_count_str(value: int, precision: int = 2) -> str:
    assert value >= 0
    labels = [" ", "K", "M", "B", "T"]
    num_digits = int(np.floor(np.log10(value)) + 1 if value > 0 else 1)
    num_groups = int(np.ceil(num_digits / 3))
    num_groups = min(num_groups, len(labels))  # don't abbreviate beyond trillions
    shift = -3 * (num_groups - 1)
    value = value * (10**shift)
    index = num_groups - 1
    rem = value - int(value)
    if precision > 0 and rem > 0.01:
        fmt = f"{{:.{precision}f}}"
        rem_str = fmt.format(rem).lstrip("0")
    else:
        rem_str = ""
    return f"{int(value):,d}{rem_str} {labels[index]}"


def to_scientific_str(value, precision: int = 1, capitalize: bool = False) -> str:
    """
    0.0015 -> "1.5e-3"
    """
    if value == 0:
        return "0"
    return f"{value:.{precision}e}".replace("e-0", "E-" if capitalize else "e-")


def print_str(*args, **kwargs):
    """
    Same as print() signature but returns a string
    """
    sstream = io.StringIO()
    kwargs.pop("file", None)
    print(*args, **kwargs, file=sstream)
    return sstream.getvalue()


def fstring(fmt_str, **kwargs):
    """
    Simulate python f-string but without `f`
    """
    locals().update(kwargs)
    return eval("f" + shlex.quote(fmt_str))


def get_format_keys(fmt_str):
    keys = []
    for literal, field_name, fmt_spec, conversion in string.Formatter().parse(fmt_str):
        if field_name:
            keys.append(field_name)
    return keys


def get_timestamp(milli_precision: int = 3):
    fmt = "%y-%m-%d %H:%M:%S"
    if milli_precision > 0:
        fmt += ".%f"
    stamp = datetime.now().strftime(fmt)
    if milli_precision > 0:
        stamp = stamp[:-milli_precision]
    return stamp


def pretty_repr_str(obj, **kwargs):
    """
    Useful to produce __repr__()
    """
    if isinstance(obj, str):
        cls_name = obj
    else:
        cls_name = obj.__class__.__name__
    kw_strs = [k + "=" + pprint.pformat(v, indent=2, compact=True) for k, v in kwargs.items()]
    new_line = len(cls_name) + sum(len(kw) for kw in kw_strs) > 84
    if new_line:
        kw = ",\n".join(kw_strs)
        return f"{cls_name}(\n{textwrap.indent(kw, '  ')}\n)"
    else:
        kw = ", ".join(kw_strs)
        return f"{cls_name}({kw})"


def pprint_(*objs, **kwargs):
    """
    Use pprint to format the objects
    """
    print(
        *[pprint.pformat(obj, indent=2) if not isinstance(obj, str) else obj for obj in objs],
        **kwargs,
    )


def get_exception_info(to_str: bool = False):
    """
    Returns:
        {'type': ExceptionType, 'value': ExceptionObject, 'trace': <traceback str>}
    """
    typ_, value, trace = sys.exc_info()
    return {
        "type": typ_.__name__ if to_str else typ_,
        "value": str(value) if to_str else value,
        "trace": "".join(traceback.format_exception(typ_, value, trace)),
    }


class DebugPrinter:
    """
    Debug print, usage: dprint = DebugPrint(enabled=True)
    dprint(...)
    """

    def __init__(self, enabled, tensor_summary: Literal["shape", "shape+dtype", "none"] = "shape"):
        """
        Args:
            tensor_summary:
                - shape: only prints shape
                - shape+dtype: also prints dtype and device
                - none: print full tensor
        """
        self.enabled = enabled
        assert tensor_summary in ["shape", "shape+dtype", "none"]
        self.tensor_summary = tensor_summary

    def __call__(self, *args, **kwargs):
        if not self.enabled:
            return
        args = [self._process_arg(a) for a in args]
        pprint_(*args, **kwargs)

    def _process_arg(self, arg):
        import numpy as np
        import torch

        if torch.is_tensor(arg):
            if self.tensor_summary == "shape":
                return str(list(arg.size()))
            elif self.tensor_summary == "shape+dtype":
                return f"{arg.dtype}{list(arg.size())}|{arg.device}"
        elif isinstance(arg, np.ndarray):
            if self.tensor_summary == "shape":
                return str(list(arg.shape))
            elif self.tensor_summary == "shape+dtype":
                return f"{arg.dtype}{list(arg.shape)}"
        return arg


@meta_decorator
def watch(func, seconds: int = 5, max_times: int = 0, keep_returns: bool = False):
    """
    Decorator: executes a function repeated with the args and
        emulate `watch -n` capability

    See `gpustat` repo: https://github.com/wookayin/gpustat/pull/41/files

    Args:
        max_times: watch for `max_times` and then exit. If 0, never exits
        keep_returns: if True, will keep the return value from the function
            and return as a list at the end
    """
    from blessings import Terminal

    def _wrapped(*args, **kwargs):
        term = Terminal()
        N = 0
        returns = []
        with term.fullscreen():
            while True:
                try:
                    with term.location(0, 0):
                        ret = func(*args, **kwargs)
                        print(term.clear_eos, end="")
                        if keep_returns:
                            returns.append(ret)
                    N += 1
                    if max_times > 0 and N >= max_times:
                        break
                    time.sleep(seconds)
                except KeyboardInterrupt:
                    break
        return returns

    return _wrapped


class PrintRedirection(object):
    """
    Context manager: temporarily redirects stdout and stderr
    """

    def __init__(self, stdout=None, stderr=None):
        """
        Args:
          stdout: if None, defaults to sys.stdout, unchanged
          stderr: if None, defaults to sys.stderr, unchanged
        """
        if stdout is None:
            stdout = sys.stdout
        if stderr is None:
            stderr = sys.stderr
        self._stdout, self._stderr = stdout, stderr

    def __enter__(self):
        self._old_out, self._old_err = sys.stdout, sys.stderr
        self._old_out.flush()
        self._old_err.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.flush()
        # restore the normal stdout and stderr
        sys.stdout, sys.stderr = self._old_out, self._old_err

    def flush(self):
        "Manually flush the replaced stdout/stderr buffers."
        self._stdout.flush()
        self._stderr.flush()


class PrintToFile(PrintRedirection):
    """
    Print to file and save/close the handle at the end.
    """

    def __init__(self, out_file=None, err_file=None):
        """
        Args:
          out_file: file path
          err_file: file path. If the same as out_file, print both stdout
              and stderr to one file in order.
        """
        self.out_file, self.err_file = out_file, err_file
        if out_file:
            out_file = os.path.expanduser(out_file)
            self.out_file = open(out_file, "w")
        if err_file:
            err_file = os.path.expanduser(out_file)
            if err_file == out_file:  # redirect both stdout/err to one file
                self.err_file = self.out_file
            else:
                self.err_file = open(os.path.expanduser(out_file), "w")
        super().__init__(stdout=self.out_file, stderr=self.err_file)

    def __exit__(self, *args):
        super().__exit__(*args)
        if self.out_file:
            self.out_file.close()
        if self.err_file:
            self.err_file.close()


def PrintSuppress(no_out=True, no_err=False):
    """
    Args:
      no_out: stdout writes to sys.devnull
      no_err: stderr writes to sys.devnull
    """
    out_file = os.devnull if no_out else None
    err_file = os.devnull if no_err else None
    return PrintToFile(out_file=out_file, err_file=err_file)


class PrintString(PrintRedirection):
    """
    Redirect stdout and stderr to strings.
    """

    def __init__(self):
        self.out_stream = io.StringIO()
        self.err_stream = io.StringIO()
        super().__init__(stdout=self.out_stream, stderr=self.err_stream)

    def stdout(self):
        "Returns: stdout as one string."
        return self.out_stream.getvalue()

    def stderr(self):
        "Returns: stderr as one string."
        return self.err_stream.getvalue()

    def stdout_by_line(self):
        "Returns: a list of stdout line by line, ignore trailing blanks"
        return self.stdout().rstrip().split("\n")

    def stderr_by_line(self):
        "Returns: a list of stderr line by line, ignore trailing blanks"
        return self.stderr().rstrip().split("\n")


# ==================== Logging filters ====================
class ExcludeLoggingFilter(logging.Filter):
    """
    Usage: logging.getLogger('name').addFilter(
        ExcludeLoggingFilter(['info mess*age', 'Warning: *'])
    )
    Supports wildcard.
    https://relaxdiego.com/2014/07/logging-in-python.html
    """

    def __init__(self, patterns):
        super().__init__()
        self._patterns = patterns

    def filter(self, record):
        if match_patterns(record.msg, include=self._patterns):
            return False
        else:
            return True


class ReplaceStringLoggingFilter(logging.Filter):
    def __init__(self, patterns, replacer: Callable):
        super().__init__()
        self._patterns = patterns
        assert callable(replacer)
        self._replacer = replacer

    def filter(self, record):
        if match_patterns(record.msg, include=self._patterns):
            record.msg = self._replacer(record.msg)


def logging_exclude_pattern(
    logger_name,
    patterns: Union[str, list[str], Callable, list[Callable], None],
):
    """
    Args:
        patterns: see groot.vla.common.utils.misc_utils.match_patterns
    """
    logging.getLogger(logger_name).addFilter(ExcludeLoggingFilter(patterns))


def logging_replace_string(
    logger_name,
    patterns: Union[str, list[str], Callable, list[Callable], None],
    replacer: Callable,
):
    """
    Args:
        patterns: see groot.vla.common.utils.misc_utils.match_patterns
    """
    logging.getLogger(logger_name).addFilter(ReplaceStringLoggingFilter(patterns, replacer))
