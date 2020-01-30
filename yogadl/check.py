"""
Functions for runtime correctness checks.

These functions are similar to Python's `assert` facility, but when a check
fails, we print both the expected value and the actual value. PEDL code should
generally prefer using `check` functions over `assert`.

The module contains similar functionality under two different naming schemes.
The first is for code that imports function symbols directly, e.g.,

  from check import check_true, check_false
  ...
  check_true(...)


The second scheme is intended for code that only imports the check module
itself.

  import check
  ...
  check.true(...)

The second scheme is generally preferred because it more clearly indicates a
module dependency at the call site and it reduces diff churn when checks are
added or removed.
"""
from typing import Any, Container, Optional, Sized, Tuple, Type, Union


class CheckFailedError(Exception):
    pass


def true(val: bool, reason: Optional[str] = None) -> None:
    if val:
        return

    msg = "CHECK FAILED! Got {}, expected True".format(val)
    if reason is not None:
        msg += ": {}".format(reason)

    raise CheckFailedError(msg)


def false(val: bool, reason: Optional[str] = None) -> None:
    if not val:
        return

    msg = "CHECK FAILED! Got {}, expected False".format(val)
    if reason is not None:
        msg += ": {}".format(reason)

    raise CheckFailedError(msg)


def is_none(val: Optional[Any], reason: Optional[str] = None) -> None:
    if val is None:
        return

    msg = "CHECK FAILED! Got {}, expected None".format(val)
    if reason is not None:
        msg += ": {}".format(reason)

    raise CheckFailedError(msg)


def is_not_none(val: Optional[Any], reason: Optional[str] = None) -> None:
    if val is not None:
        return

    msg = "CHECK FAILED! Got {}, expected non-None".format(val)
    if reason is not None:
        msg += ": {}".format(reason)

    raise CheckFailedError(msg)


def eq(val: Any, expected: Any, reason: Optional[str] = None) -> None:
    if val == expected:
        return

    msg = "CHECK FAILED! Got {}, expected {}".format(val, expected)
    if reason is not None:
        msg += ": {}".format(reason)

    raise CheckFailedError(msg)


def not_eq(x: Any, y: Any, reason: Optional[str] = None) -> None:
    if x != y:
        return

    msg = "CHECK FAILED! Got {}, expected value != {}".format(x, y)
    if reason is not None:
        msg += ": {}".format(reason)

    raise CheckFailedError(msg)


def gt(x: Any, y: Any, reason: Optional[str] = None) -> None:
    if x > y:
        return

    msg = "CHECK FAILED! Got {}, expected value > {}".format(x, y)
    if reason is not None:
        msg += ": {}".format(reason)

    raise CheckFailedError(msg)


def gt_eq(x: Any, y: Any, reason: Optional[str] = None) -> None:
    if x >= y:
        return

    msg = "CHECK FAILED! Got {}, expected value >= {}".format(x, y)
    if reason is not None:
        msg += ": {}".format(reason)

    raise CheckFailedError(msg)


def lt(x: Any, y: Any, reason: Optional[str] = None) -> None:
    if x < y:
        return

    msg = "CHECK FAILED! Got {}, expected value < {}".format(x, y)
    if reason is not None:
        msg += ": {}".format(reason)

    raise CheckFailedError(msg)


def lt_eq(x: Any, y: Any, reason: Optional[str] = None) -> None:
    if x <= y:
        return

    msg = "CHECK FAILED! Got {}, expected value <= {}".format(x, y)
    if reason is not None:
        msg += ": {}".format(reason)

    raise CheckFailedError(msg)


def equal_lengths(x: Sized, y: Sized, reason: Optional[str] = None) -> None:
    if len(x) == len(y):
        return

    msg = "CHECK FAILED! Expected lengths {} and {} to be equal".format(len(x), len(y))
    msg += "; values: {}, {}".format(x, y)
    if reason is not None:
        msg += ": {}".format(reason)

    raise CheckFailedError(msg)


def len_eq(val: Sized, expected_len: int, reason: Optional[str] = None) -> None:
    val_len = len(val)
    if val_len == expected_len:
        return

    msg = "CHECK FAILED! Got length {}, expected length {}".format(val_len, expected_len)
    if reason is not None:
        msg += ": {}".format(reason)
    msg += ". Values: {}".format(val)

    raise CheckFailedError(msg)


def is_in(val: Any, expected: Container[Any], reason: Optional[str] = None) -> None:
    if val in expected:
        return

    msg = "CHECK FAILED! "

    # Report a more concise error message when `expected` is a dict.
    if isinstance(expected, dict):
        msg += "'{}' is not in {}".format(val, list(expected.keys()))
    else:
        msg += "'{}' is not in {}".format(val, expected)

    if reason is not None:
        msg += ": {}".format(reason)

    raise CheckFailedError(msg)


def not_in(val: Any, expected: Container[Any], reason: Optional[str] = None) -> None:
    if val not in expected:
        return

    msg = "CHECK FAILED! Got {}, expected value not in {}".format(val, expected)
    if reason is not None:
        msg += ": {}".format(reason)

    raise CheckFailedError(msg)



def is_type(val: Any, expected: type, reason: Optional[str] = None) -> None:
    if type(val) == expected:
        return

    msg = "CHECK FAILED! {} has type {}, expected type {}".format(val, type(val), expected)
    if reason is not None:
        msg += ": {}".format(reason)

    raise CheckFailedError(msg)


def is_instance(
    val: Any, expected: Union[Type, Tuple[Type, ...]], reason: Optional[str] = None
) -> None:
    if isinstance(val, expected):
        return

    msg = "CHECK FAILED! {} has type {}, expected isinstance of {}".format(val, type(val), expected)
    if reason is not None:
        msg += ": {}".format(reason)

    raise CheckFailedError(msg)


def is_not_instance(
    val: Any, expected: Union[Type, Tuple[Type, ...]], reason: Optional[str] = None
) -> None:
    if not isinstance(val, expected):
        return

    msg = "CHECK FAILED! {} has type {}, expected not isinstance of {}".format(
        val, type(val), expected
    )
    if reason is not None:
        msg += ": {}".format(reason)

    raise CheckFailedError(msg)


def is_subclass(val: Any, expected: type, reason: Optional[str] = None) -> None:
    if issubclass(val, expected):
        return

    msg = "CHECK FAILED! {} has type {}, expected issubclass of {}".format(val, type(val), expected)
    if reason is not None:
        msg += ": {}".format(reason)

    raise CheckFailedError(msg)
