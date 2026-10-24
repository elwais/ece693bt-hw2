"""A modified dataclass that removes boilerplate associated with making custom `nn.Module`s."""

import inspect
from dataclasses import MISSING, Field, dataclass, field, fields
from types import CodeType
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar, cast

import torch.nn as nn
from typing_extensions import dataclass_transform

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

M = TypeVar("M", bound=nn.Module)
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


class Factory(Protocol[T_co]):
    """A zero-argument function or class that builds a value.

    Requiring ``__code__`` is what excludes modules: an ``nn.Module`` is
    callable, so it satisfies ``Callable[[], T]``, but it has no ``__code__``.
    Lambdas and ``def`` functions do, and so (to the type checker) do classes,
    so ``built(nn.Sequential)`` is accepted. ``functools.partial`` and other
    callable objects are rejected; wrap them in a lambda.
    """

    __code__: CodeType

    def __call__(self) -> T_co: ...

# Marks "argument not supplied". A private object, so None stays a valid value.
_UNSET: object = object()


def _has_default(f: Field[object]) -> bool:
    return f.default is not MISSING or f.default_factory is not MISSING


def _is_kw_only(f: Field[object]) -> bool:
    # Field.kw_only is typed bool | MISSING; after dataclass() it is always a bool.
    return f.kw_only is True


def _default_of(f: Field[object]) -> object:
    if f.default is not MISSING:
        return f.default
    return f.default_factory() if f.default_factory is not MISSING else None


def _require_module_class(obj: object) -> type[nn.Module]:
    """Runtime guard for untyped callers: accept only nn.Module subclasses."""
    if not (isinstance(obj, type) and issubclass(obj, nn.Module)):
        raise TypeError(
            f"@nn_dataclass can only decorate nn.Module subclasses, got {obj!r}"
        )
    return obj


def built(default_factory: Factory[T], *, init: Literal[False] = False) -> T:
    """Declare a field that the class builds itself, rather than taking as an argument.

    Shorthand for ``field(init=False, default_factory=default_factory)``. Use it
    for fixed architecture, as opposed to settings passed to the constructor::

        head: nn.Sequential = built(lambda: nn.Sequential(nn.Linear(128, 10)))

    Pass a factory (e.g. a ``lambda``), not a module. ``default_factory`` is
    called once per instance, so each instance gets its own module and weights.
    A module created inline in the class body would be created once and shared
    by every instance.

    The parameter names are part of the contract with ``dataclass_transform``:
    ``default_factory`` tells the type checker the field has a default, and
    ``init: Literal[False] = False`` tells it the field is not an ``__init__``
    parameter. Do not pass ``init``; it exists only for the type checker.

    Passing a module is a type error: ``default_factory`` is a ``Factory``,
    which a module does not satisfy (it has no ``__code__``).

    Raises:
        TypeError: if ``default_factory`` is an ``nn.Module``. This repeats
            the type check at runtime, for callers that are not type-checked.
    """
    if isinstance(default_factory, nn.Module):
        raise TypeError(
            "built() takes a factory, not a module: "
            + f"write built(lambda: {type(default_factory).__name__}(...))"
        )
    return field(init=init, default_factory=default_factory)


@dataclass_transform(eq_default=False, field_specifiers=(field, built))
def nn_dataclass(cls: type[M]) -> type[M]:
    """Give an nn.Module subclass dataclass-style fields and a generated __init__.

    Fields are declared as class annotations and bound by name, as with
    ``@dataclass``. The generated ``__init__`` calls ``nn.Module.__init__``
    first, so assigned submodules and Parameters register normally. A
    ``__post_init__`` method, if defined, runs last.

    Field specifiers: ``dataclass_transform`` lists ``field`` and ``built`` in
    ``field_specifiers``. Without that, the type checker treats
    ``x: T = field(...)`` as a plain default value and ignores the arguments
    to ``field``. ``field(init=False, ...)`` would then still appear as an
    ``__init__`` parameter to the type checker, while the runtime rejects it
    with ``TypeError``. Listing ``field`` makes the checker honor ``init``,
    ``kw_only``, ``default`` and ``default_factory``, so the static signature
    matches the generated one. Only functions in this tuple are recognized:
    any other helper that wraps ``field`` is treated as a plain default, which
    is why ``built`` is listed alongside ``field`` (see PEP 681, "Field
    specifiers").

    Limitations: ``cls`` must subclass ``nn.Module`` directly, because the
    generated ``__init__`` calls ``nn.Module.__init__`` and skips any
    intermediate base ``__init__``. ``cls`` must not define its own
    ``__init__``.

    Raises:
        TypeError: if ``cls`` is not an nn.Module subclass, or defines
            ``__init__`` itself.
    """
    _ = _require_module_class(cls)
    if "__init__" in vars(cls):
        raise TypeError(
            f"{cls.__qualname__} defines __init__; @nn_dataclass generates it"
        )

    # init=False: we generate __init__ ourselves. dataclass() modifies cls in
    # place, so the return value is not needed.
    had_doc = cls.__doc__ is not None
    _ = dataclass(cls, init=False, eq=False, repr=False)
    dc = cast("type[DataclassInstance]", cls)
    cls_fields: tuple[Field[object], ...] = fields(dc)

    # dataclass() writes an auto-docstring from the signature of cls.__init__,
    # which is still nn.Module's (*args, **kwargs) at this point. Rebuild it
    # from the fields so help(cls) shows the real parameters.
    if not had_doc:
        doc_params = [
            inspect.Parameter(
                f.name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=_default_of(f) if _has_default(f) else inspect.Parameter.empty,
                annotation=f.type,
            )
            for f in cls_fields
            if f.init
        ]
        cls.__doc__ = f"{cls.__name__}{inspect.Signature(doc_params)}"

    # One signature for all fields with init=True, built once. Every parameter
    # defaults to _UNSET, so Signature.bind never rejects a field that has its
    # own default; the real default is resolved in custom_init.
    # Positional fields must precede keyword-only ones, as dataclass orders them.
    init_fields = [f for f in cls_fields if f.init]
    init_fields.sort(key=_is_kw_only)
    signature = inspect.Signature(
        [
            inspect.Parameter(
                f.name,
                inspect.Parameter.KEYWORD_ONLY
                if _is_kw_only(f)
                else inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=_UNSET,
            )
            for f in init_fields
        ]
    )
    cls_name = cls.__qualname__

    def custom_init(self: nn.Module, *args: object, **kwargs: object) -> None:
        # Must run before any attribute assignment so PyTorch's __setattr__
        # hook can register modules and parameters.
        nn.Module.__init__(self)

        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = cast("dict[str, object]", bound.arguments)

        for f in cls_fields:
            value: object
            if f.init and arguments[f.name] is not _UNSET:
                value = arguments[f.name]
            elif _has_default(f):
                value = _default_of(f)
            elif f.init:
                raise TypeError(
                    f"{cls_name}.__init__() missing required argument: {f.name!r}"
                )
            else:
                # init=False with no default: the user sets it in __post_init__.
                continue
            setattr(self, f.name, value)

        post_init = getattr(self, "__post_init__", None)
        if callable(post_init):
            _ = post_init()

    setattr(cls, "__init__", custom_init)
    return cls
