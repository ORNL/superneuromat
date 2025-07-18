"""Utility functions for SuperNeuroMAT.

.. currentmodule:: superneuromat.util

.. autofunction:: getenvbool
.. autofunction:: is_intlike
.. autofunction:: pretty_spike_train
.. autofunction:: print_spike_train

"""
from __future__ import annotations
import os

import numpy as np

# typing
from typing import Any, TYPE_CHECKING


def getenv(key, default=None):
    """Get the value of an environment variable or return a default."""
    s = os.environ.get(key, default)
    if isinstance(s, str):
        s = s.lower().strip()
        if s == '':
            return default
    return s


if TYPE_CHECKING:
    from typing import overload, Literal
    @overload
    def getenvbool(key: str, default: bool | None | str | Any = None, force_bool: Literal[False] = False) -> Any: ...
    @overload
    def getenvbool(key: str, default: bool | None | str = None, force_bool: Literal[False] = False) -> bool | str: ...
    @overload
    def getenvbool(key: str, default: bool | None = None, force_bool: Literal[True] = True) -> bool: ...


def getenvbool(key, default=None, force_bool=False):
    """Get the value of an environment variable and attempt to cast to bool, or return a default.

    Parameters
    ----------
    key : str
        The name of the environment variable.
    default : bool | None, optional
        The default value to return if the environment variable is not set.
    force_bool : bool, default=False
        If True, raise an error if the environment variable is not a bool.

    Returns
    -------
    bool | str
        The value of the environment variable, or the default value if the environment variable is not set.

    Raises
    ------
    ValueError
        If the environment variable is not a bool and force_bool is ``True``, or
        if you set ``force_bool=True`` and a non-bool value for ``default``
    """
    s = os.environ.get(key, default)
    if isinstance(s, str):
        s2 = s.lower().strip()
        if s2 in ('false', '0'):
            return False
        elif s2 in ('true', '1'):
            return True
        if s == '':
            return default
    if force_bool and not isinstance(s, bool):
        if not isinstance(default, bool):
            msg = f"Expected either True|False as default for getenvbool() {key}"
            msg += f" because you set force_bool=True, but you set default={default!r}"
        else:
            msg = f"Expected either true|false for environment variable {key}: {s}"
        raise ValueError(msg)
    return s


def is_intlike(x):
    """Returns ``True`` if ``x`` is equivalent to an integer.

    Raises
    ------
    ValueError
        If ``x`` cannot be coerced to an integer.
    """
    if isinstance(x, int):
        return True
    else:
        return x == int(x)


def is_intlike_catch(x):
    """Returns ``True`` if ``x`` is equivalent to an integer or False if x is not a number."""
    if isinstance(x, int):
        return True
    else:
        try:
            return x == int(x)
        except (ValueError, TypeError):
            return False


def int_err(x, name='', fname='', msg=None):
    """Cast int and raise a ValueError if x is not an int or int-like."""
    try:
        y = int(x)
    except ValueError as e:
        if msg is None:
            msg = f"{ f'{fname} c' if fname else 'C'}ould not convert argument {name} of type {type(x)} to int: {x!r}"
        raise ValueError(msg) from e
    except TypeError as e:
        if msg is None:
            msg = f"{fname} argument {name} of type {type(x)} must be a string, bytes-like object, or a real number, not {x!r}"
        raise TypeError(msg) from e
    if y != x:
        if msg is None:
            msg = f"{fname} casting argument {name} to int resulted in loss of precision: {y} <- {x!r}"
        raise ValueError(msg)
    return y


def float_err(x, name='', fname='', msg=None):
    """Cast float and raise a custom error"""
    try:
        return float(x)
    except ValueError as e:
        if msg is None:
            msg = f"{ f'{fname} c' if fname else 'C'}ould not convert argument {name} of type {type(x)} to float: {x!r}"
        raise ValueError(msg) from e
    except TypeError as e:
        if msg is None:
            msg = f"{fname} argument {name} of type {type(x)} must be a string or a real number, not {x!r}"
        raise TypeError(msg) from e


def accessor_slice(s: slice[Any, Any, Any]) -> slice:
    if not isinstance(s, slice):
        msg = f"Expected slice, but received {type(s)}"
        raise TypeError(msg)
    start = int(s.start) if s.start is not None else None
    stop = int(s.stop) if s.stop is not None else None
    step = int(s.step) if s.step is not None else None
    return slice(start, stop, step)


def slice_indices(s: slice[Any, Any, Any], max_len: int = 0) -> list[int]:
    s = accessor_slice(s)
    if s.step is None or s.step >= 0 or s.stop is None:
        stop = max_len if s.stop is None else min(s.stop, max_len)
    else:  # step is negative, stop is not None. Ignore stop
        stop = max_len
    return list(range(stop))[s]


def pretty_spike_train(
        spike_train: list[list[bool]] | list[np.ndarray] | np.ndarray,
        max_steps: int | None = 11,
        max_neurons: int | None = 28,
        use_unicode: bool | Any = True,
    ):
    """Prints the spike train.

    Parameters
    ----------
    spike_train: list[list[bool]] | list[np.ndarray] | np.ndarray
        The spike train to show.
    max_steps : int | None, optional
        Limits the number of steps which will be included.
        If limited, only a total of ``max_steps`` first and last steps will be included.
    max_neurons : int | None, optional
        Limits the number of neurons which will be included.
        If limited, only a total of ``max_neurons`` first and last neurons will be included.
    use_unicode : bool, default=True
        If ``True``, use unicode characters to represent spikes.
        Otherwise fallback to ascii characters.
    """
    lines = []
    steps = len(spike_train)
    neurons = len(spike_train[0]) if steps else 0
    t_nchar = len(str(steps - 1))
    i_nchar = max(len(str(neurons - 1)), 2)  # should be at least 2 wide
    c0 = f"{'│ ':<{i_nchar}}" if use_unicode else f"{'0 ':<{i_nchar}}"
    c1 = f"{'├─':<{i_nchar}}" if use_unicode else f"{'1 ':<{i_nchar}}"
    sep = '' if use_unicode else ''
    ellip = '…' if use_unicode else '.'
    vellip = '⋮' if use_unicode else '.'

    horizontally_continuous = max_neurons is None or neurons <= max_neurons

    def spiked_str(spiked):
        if horizontally_continuous:
            return sep.join([c1 if x else c0 for x in spiked])
        else:
            fi = max_neurons // 2  # pyright: ignore[reportOptionalOperand]
            li = max_neurons // 2  # pyright: ignore[reportOptionalOperand]
            first = spiked[:fi]
            last = spiked[-li:]
            return sep.join([c1 if x else c0 for x in first] + [ellip] + [c1 if x else c0 for x in last])

    # print header
    if horizontally_continuous:
        ids = [f"{i:<{i_nchar}d}" for i in range(neurons)]
    else:
        fi = max_neurons // 2  # pyright: ignore[reportOptionalOperand]
        first = [f"{i:<{i_nchar}d}" for i in range(fi)]
        last = [f"{i:<{i_nchar}d}" for i in range(neurons - fi, neurons)]
        ids = first + [ellip] + last
    lines.append(f"{'t':>{t_nchar}s}|id{sep.join(ids)} ")

    if max_steps is None or len(spike_train) <= max_steps:
        for time, spiked in enumerate(spike_train):
            lines.append(f"{time:>{t_nchar}d}: [{spiked_str(spiked)}]")
    else:
        fi = max_steps // 2
        li = max_steps // 2
        if max_neurons is None:
            max_neurons = 0
        first = spike_train[:fi]
        last = spike_train[-li:]
        for time, spiked in enumerate(first):
            lines.append(f"{time:>{t_nchar}d}: [{spiked_str(spiked)}]")
        lines.append(f"{'.' * t_nchar}  [{vellip * max(len(spike_train[0]), max_neurons)}]")
        for time, spiked in enumerate(last):
            lines.append(f"{time + steps - fi:>{t_nchar}d}: [{spiked_str(spiked)}]")
    return lines


def print_spike_train(spike_train, max_steps=None, max_neurons=None, use_unicode=True):
    """Prints the spike train.

    Parameters
    ----------
    spike_train: list[list[bool]] | list[np.ndarray] | np.ndarray
        The spike train to print.
    max_steps : int | None, optional
        Limits the number of steps which will be printed.
        If limited, only a total of ``max_steps`` first and last steps will be printed.
    max_neurons : int | None, optional
        Limits the number of neurons which will be printed.
        If limited, only a total of ``max_neurons`` first and last neurons will be printed.
    use_unicode : bool, default=True
        If ``True``, use unicode characters to represent spikes.
        Otherwise fallback to ascii characters.
    """
    print('\n'.join(pretty_spike_train(spike_train, max_steps, max_neurons, use_unicode)))
