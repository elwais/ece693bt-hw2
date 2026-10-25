"""Checked narrowing of a module's output to Tensor; a runtime-checked cast"""

from typing import Any

from torch import Tensor


def as_tensor(x: Any) -> Tensor:  # pyright: ignore[reportExplicitAny, reportAny]
    """Return `x` typed as a Tensor, raising TypeError if it is anything else.

    nn.Module.__call__ is typed to return Any, because a forward may return a
    tuple (nn.LSTM), a dict, or whatever a forward hook substitutes. Wrapping a
    module call in as_tensor turns that Any into Tensor for the type checker
    and fails at the call that produced the wrong value, not at a later use.

    The parameter is Any, not object, so that call sites passing a module's
    output are not flagged by reportAny; the ignores stay in this one place.
    """
    if not isinstance(x, Tensor):
        name = type(x).__name__  # pyright: ignore[reportAny]
        raise TypeError(f"expected a Tensor, got {name}")
    return x
