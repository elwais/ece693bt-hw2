# `nn_dataclass`: dataclass ergonomics for `nn.Module`

This document covers `nn_dataclass`, the decorator in `utils/nn_dataclass.py`. It is in three parts:

1. **Why and how to use it.** A before/after example, then the rules.
2. **Why plain `@dataclass` does not work** on `nn.Module`, with the failures reproduced.
3. **How it is built.** The type-hint machinery (`TypeVar`, `dataclass_transform`, `DataclassInstance`) and a walkthrough of the code that keeps `nn.Module` internals intact.

All outputs below were produced by running the code in this repository with PyTorch 2.14.1 on Python 3.12.

---

## 1. Motivation: before and after

Consider a small CNN classifier in the spirit of Setup Task 0, with a few hyperparameters and four layer groups. Written the usual way, the module repeats each hyperparameter three times (parameter list, `self.x = x`, and the layer that uses it):

```python
import torch
import torch.nn as nn


class TinyNetBare(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        width: int = 16,
        hidden: int = 128,
        n_classes: int = 10,
        p_drop: float = 0.25,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.width = width
        self.hidden = hidden
        self.n_classes = n_classes
        self.p_drop = p_drop

        self.conv1 = nn.Conv2d(in_ch, width, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, 2 * width, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(2 * width)
        self.pool = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(p_drop)
        self.fc1 = nn.Linear(2 * width * 12 * 12, hidden)
        self.drop2 = nn.Dropout(p_drop)
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.drop1(self.pool(x)).flatten(1)
        x = self.drop2(torch.relu(self.fc1(x)))
        return self.fc2(x)
```

With `nn_dataclass`, the hyperparameters are declared once as annotated fields. The constructor, the `self.x = x` assignments and the type hints are generated for you. Submodules are created in `__post_init__`, which runs after the generated `__init__` has bound the fields:

```python
import torch
import torch.nn as nn

from utils import nn_dataclass


@nn_dataclass
class TinyNet(nn.Module):
    in_ch: int = 1
    width: int = 16
    hidden: int = 128
    n_classes: int = 10
    p_drop: float = 0.25

    def __post_init__(self) -> None:
        w = self.width
        self.conv1 = nn.Conv2d(self.in_ch, w, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(w)
        self.conv2 = nn.Conv2d(w, 2 * w, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(2 * w)
        self.pool = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(self.p_drop)
        self.fc1 = nn.Linear(2 * w * 12 * 12, self.hidden)
        self.drop2 = nn.Dropout(self.p_drop)
        self.fc2 = nn.Linear(self.hidden, self.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.drop1(self.pool(x)).flatten(1)
        x = self.drop2(torch.relu(self.fc1(x)))
        return self.fc2(x)
```

The two versions are the same network. Verified in one process:

```
same state_dict keys+shapes: True | n tensors: 18
param count bare/dec: 596138 596138
forward shapes: torch.Size([2, 10]) torch.Size([2, 10])
```

The decorated class body is 25 lines against 32 for the bare version. The gain is less about line count than about having a single source of truth for each hyperparameter, and a constructor signature that is accurate both at runtime and for the type checker (section 4.1).

Here is what `help()` shows for the decorated class:

```
TinyNet(in_ch: int = 1, width: int = 16, hidden: int = 128, n_classes: int = 10, p_drop: float = 0.25)
```

---

## 2. Using it

### 2.1 Rules

| Rule | Why |
|---|---|
| Apply only to a direct subclass of `nn.Module`. | Checked at runtime. `TypeError` otherwise. |
| Do not define `__init__`. The decorator generates it. | Checked at runtime. `TypeError` if you do, so the generated `__init__` never silently replaces yours. |
| Declare hyperparameters as annotated class fields, with defaults where useful. | These become the constructor signature, as with `@dataclass`. |
| Create submodules and Parameters in `__post_init__`. | The generated `__init__` calls `nn.Module.__init__` first, so assignments in `__post_init__` register normally. |
| Do not annotate a submodule at class level. | An annotated field becomes a constructor argument (see 2.3). |
| Subclass with care. | See 2.4. |

### 2.2 Constructor behaviour

The generated constructor behaves like a dataclass `__init__`:

```python
TinyNet()                      # all defaults
TinyNet(3, 32)                 # positional, in field order
TinyNet(hidden=64, p_drop=0.5) # keyword
TinyNet(width=8, nope=1)       # TypeError: got an unexpected keyword argument 'nope'
```

Field defaults and `default_factory` are honoured. A missing required argument raises, for example `TypeError: Block.__init__() missing required argument: 'width'` for a class whose `width` has no default.

### 2.3 Class-level submodules become constructor arguments

If you annotate a submodule at class level, it is a dataclass field, so the caller must supply it:

```python
@nn_dataclass
class Bad(nn.Module):
    head: nn.Linear

Bad(head=nn.Linear(2, 2))  # works
Bad()                      # TypeError: Bad.__init__() missing required argument: 'head'
```

This is sometimes what you want (dependency injection). For layers the module owns, create them in `__post_init__`.

### 2.4 Subclassing

A decorated class is an ordinary class, so subclasses work, but a subclass's own annotated fields are **not** added to the constructor. The generated `__init__` is inherited unchanged:

```python
class Child(TinyNet):
    extra: int = 9

Child(width=3).extra   # 9, a class attribute, not set by any constructor
Child(extra=5)         # TypeError: got an unexpected keyword argument 'extra'
```

Treat a decorated class as a final constructor. Put shared behaviour in a non-decorated base class, or decorate the subclass too, which replaces its inherited `__init__`. If you go that route, check that the subclass's own `__post_init__` calls `super().__post_init__()` where it needs the parent's setup.

---

## 3. Why plain `@dataclass` does not suffice

`dataclass` is designed for plain objects. Applied to an `nn.Module` it breaks the module in four ways. Each was reproduced:

```python
from dataclasses import dataclass
import torch.nn as nn

@dataclass
class Bare(nn.Module):
    width: int
    def __post_init__(self) -> None:
        self.proj = nn.Linear(self.width, self.width)

Bare(4)
# AttributeError: cannot assign module before Module.__init__() call
```

### 3.1 The generated `__init__` never calls `nn.Module.__init__`

`dataclass` writes its own `__init__`. It assigns the fields and then runs `__post_init__`, but nothing calls `super().__init__()`. `nn.Module.__setattr__` relies on the bookkeeping dictionaries that `nn.Module.__init__` creates (`_parameters`, `_modules`, `_buffers`). Before that call, assigning a submodule or a `Parameter` raises. Plain `int` fields assign without error, so the failure appears only once a submodule or `Parameter` is assigned.

### 3.2 `eq=True` makes instances unhashable and compares by field value

`dataclass` defaults to `eq=True`. That sets `__eq__` to compare fields, and sets `__hash__` to `None`:

```
bare dataclass: Cfg(5)==Cfg(5): True
modules() error: TypeError unhashable type: 'Cfg'
```

Two different modules with equal hyperparameters compare equal, which is wrong: they hold different parameters. Worse, `nn.Module` uses modules as set members internally. `modules()` (and `named_modules()`, which it wraps) keeps a memo set, so it crashes on the unhashable instance.

### 3.3 `repr=True` replaces the module printout

`dataclass` generates `__repr__` from the fields. `nn.Module.__repr__` prints the layer tree, which is what you use to inspect a network. The generated version drops it.

### 3.4 The auto-docstring is built from the wrong signature

When a class has no docstring, `dataclass` writes `__doc__` from `inspect.signature(cls)`. At that point `cls.__init__` is still `nn.Module`'s `(*args, **kwargs)`, so `help()` would show `TinyNet(*args: Any, **kwargs: Any)`. `nn_dataclass` rebuilds the docstring after the fields are known (section 4.5).

### 3.5 What `nn_dataclass` does instead

| Attribute | Plain `@dataclass` | `nn_dataclass` |
|---|---|---|
| `__init__` | generated; no `nn.Module.__init__` call | generated; calls `nn.Module.__init__` first |
| `__eq__` | generated (compares fields) | not generated; identity equality inherited from `object` |
| `__hash__` | set to `None` (unhashable) | inherited; modules stay hashable |
| `__repr__` | generated (fields only) | inherited; layer-tree printout kept |
| `__doc__` (no docstring) | built from the generated `__init__` | rebuilt from the fields |
| `__dataclass_fields__`, `__dataclass_params__`, `__match_args__` | set | set (harmless; `fields()` works) |
| `nn.Module` `__setattr__`, `__getattr__`, `state_dict`, hooks | untouched | untouched |

The fourth row is the point: `nn_dataclass` changes only what the dataclass machinery would otherwise overwrite, and it leaves the rest of the module alone.

---

## 4. How it is built

### 4.1 Type-hint plumbing

The decorator has to satisfy two readers: Python at runtime, and the type checker. Four typing tools do the work.

**`TypeVar` bound to `nn.Module`.**

```python
M = TypeVar("M", bound=nn.Module)

def nn_dataclass(cls: type[M]) -> type[M]: ...
```

`type[M]` means "a class object whose instances are `M`". Returning `type[M]` tells the checker that the decorated name still refers to the same class, so `TinyNet(...)` is typed as `TinyNet`. An earlier version annotated the return as `nn.Module`, which made the decorated name an instance type. The checker then rejected `type[Module]` passed to `dataclass` and `fields`.

**`dataclass_transform` (PEP 681).**

```python
@dataclass_transform(eq_default=False)
def nn_dataclass(cls: type[M]) -> type[M]: ...
```

A type checker cannot see inside a decorator. Without a hint it treats `TinyNet` as an ordinary class, so it does not know that `hidden` is a constructor argument. It then flags the class's own attributes as uninitialised, and it does not check constructor calls.

`dataclass_transform` tells the checker that this decorator behaves like `@dataclass`. The checker synthesises a constructor from the annotated fields and checks calls against it. These are the three errors the checker reports for misuse of a decorated class, from a probe file (verified):

```
Argument of type "Literal['wide']" cannot be assigned to parameter "width" of type "int" in function "__init__"
Argument missing for parameter "width"
No parameter named "extra"
```

At runtime the decorator does nothing extra with this. `typing_extensions` attaches a marker to the function, which the checker reads statically:

```
nn_dataclass.__dataclass_transform__ = {'eq_default': False, 'order_default': False, 'kw_only_default': False, 'frozen_default': False, 'field_specifiers': (), 'kwargs': {}}
```

`eq_default=False` matches the runtime `eq=False`. Without it the checker would assume `__eq__` exists.

The import is `typing_extensions`, not `typing`, because `dataclass_transform` only entered `typing` in Python 3.11 and the project supports 3.10. `typing_extensions` is already installed as a torch dependency.

**`DataclassInstance` under `TYPE_CHECKING`.**

```python
if TYPE_CHECKING:
    from _typeshed import DataclassInstance

dc = cast("type[DataclassInstance]", cls)
cls_fields: tuple[Field[object], ...] = fields(dc)
```

`dataclasses.fields()` is typed to accept only `DataclassInstance`, which is a `Protocol`. A protocol is satisfied structurally: any class with a `__dataclass_fields__` attribute matches. At runtime `cls` already has that attribute, because `dataclass()` has run, but the checker cannot see it. `cast` asserts the type without changing the object. `_typeshed` exists only for the checker, so the import sits behind `TYPE_CHECKING`, and the cast uses a string so nothing is evaluated at runtime.

**`object` instead of `Any`.** Values from `Field.default`, `bound.arguments` and `getattr` are `Any`. Strict mode reports `Any` as `reportAny`. The code annotates them `object` and narrows with `is` checks, so the checker keeps the type information. `_UNSET: object = object()` is a private sentinel: `None` is a valid user value, so it cannot mark "not supplied".

### 4.2 Runtime guards

Abridged:

```python
def _require_module_class(obj: object) -> type[nn.Module]:
    if not (isinstance(obj, type) and issubclass(obj, nn.Module)):
        raise TypeError(...)
    return obj
```

The check takes `object`, not `type[M]`. If the decorator's own parameter were `type[M]`, the checker would mark `isinstance(cls, type)` as always true and the `raise` as unreachable. Giving the guard an `object` parameter lets the checker treat the check as meaningful, while still enforcing it at runtime for untyped callers, such as code that uses `Any`, or code run from a notebook.

The second guard looks for `__init__` in the class's own `__dict__` (`vars(cls)`), not with `hasattr`. `hasattr` would also match an inherited `__init__`, which the decorator is meant to replace.

### 4.3 Applying `dataclass` in place, without its `__init__`

```python
had_doc = cls.__doc__ is not None
_ = dataclass(cls, init=False, eq=False, repr=False)
```

- `dataclass` modifies `cls` in place and returns it, so the result is discarded. The `_ =` avoids an unused-result warning.
- `init=False` stops `dataclass` writing the `__init__` that breaks `nn.Module` (3.1).
- `eq=False` stops the field-comparing `__eq__` and the `__hash__ = None` (3.2).
- `repr=False` keeps `nn.Module.__repr__` (3.3).
- `had_doc` records whether the class had a docstring, for 4.5.

Everything `dataclass` still does is useful: it collects `__annotations__` into the field list, resolves defaults and `default_factory`, rejects mutable defaults such as a bare `list`, and rejects field ordering mistakes such as a required field after a defaulted one.

### 4.4 Field collection

```python
cls_fields: tuple[Field[object], ...] = fields(dc)
```

`fields()` returns one `Field` per annotated dataclass field, in declaration order. It excludes `ClassVar` annotations. Each `Field` carries `name`, `type`, `default`, `default_factory` and `init`. The decorator uses those four attributes. The `init` flag matters: `field(init=False)` fields are set by `__post_init__` or by a default, not by the constructor.

### 4.5 Rebuilding the docstring

Abridged:

```python
if not had_doc:
    doc_params = [
        inspect.Parameter(f.name, POSITIONAL_OR_KEYWORD,
                          default=_default_of(f) if _has_default(f) else inspect.Parameter.empty,
                          annotation=f.type)
        for f in cls_fields if f.init
    ]
    cls.__doc__ = f"{cls.__name__}{inspect.Signature(doc_params)}"
```

Only classes without a docstring get the generated one, so a hand-written docstring is never overwritten. The signature is built from the real fields, so `help()` shows the true constructor. The regression test `test_docstring_lists_real_parameters` checks this.

### 4.6 The constructor signature, built once

```python
signature = inspect.Signature([
    inspect.Parameter(f.name, POSITIONAL_OR_KEYWORD, default=_UNSET)
    for f in cls_fields if f.init
])
```

The signature is built once, when the class is decorated, not on every construction. Each parameter defaults to `_UNSET`, not to the field's real default. This is deliberate. A `default_factory` field has no single default value to put in a signature, and writing `None` as the default would make "not supplied" indistinguishable from "supplied as `None`". Resolving the real default in `custom_init` keeps the factory call per-instance and keeps the two cases apart.

Using `Signature.bind` rather than hand-written argument parsing gives Python's own error messages for free: unexpected keywords, multiple values for one argument, and too many positional arguments.

### 4.7 `custom_init`: the constructor

Abridged from `utils/nn_dataclass.py`, with the lettered steps annotated:

```python
def custom_init(self: nn.Module, *args: object, **kwargs: object) -> None:
    nn.Module.__init__(self)                        # (a)

    bound = signature.bind(*args, **kwargs)         # (b)
    bound.apply_defaults()
    arguments = cast("dict[str, object]", bound.arguments)

    for f in cls_fields:                            # (c)
        value: object
        if f.init and arguments[f.name] is not _UNSET:
            value = arguments[f.name]
        elif _has_default(f):
            value = _default_of(f)
        elif f.init:
            raise TypeError(...missing required argument...)
        else:
            continue                                # init=False, no default
        setattr(self, f.name, value)                # (d)

    post_init = getattr(self, "__post_init__", None)  # (e)
    if callable(post_init):
        _ = post_init()

setattr(cls, "__init__", custom_init)               # (f)
return cls
```

**(a) `nn.Module.__init__` runs first.** This creates `_parameters`, `_modules`, `_buffers`, and the hook dictionaries, so every later assignment has somewhere to register. This is the fix for 3.1. The call names `nn.Module.__init__` explicitly rather than using `super()`. `super()` would continue the MRO after the decorated class, and the next `__init__` there is not guaranteed to be `nn.Module`'s, which is the inheritance limit in 2.4.

**(b) Binding.** `bind` maps positional and keyword arguments onto the signature. `apply_defaults` fills the remaining names with `_UNSET`.

**(c) Resolution order.** For each field, in declaration order:

1. A supplied argument wins.
2. Otherwise the field's `default` is used.
3. Otherwise `default_factory()` is called, so each instance gets a fresh object.
4. Otherwise a required `init=True` field raises a `TypeError` that names the class and field.
5. Otherwise an `init=False` field with no default is skipped, so `__post_init__` can set it.

**(d) `setattr` goes through PyTorch.** `setattr(self, name, value)` calls `nn.Module.__setattr__`, which dispatches by value type. An `nn.Module` value goes to `_modules`, an `nn.Parameter` to `_parameters`, and a registered buffer to `_buffers`. Anything else, such as an `int` or a `str`, is stored as a plain instance attribute. That is why hyperparameters and layers can share one loop.

**(e) `__post_init__`.** `getattr` with a default is used because `nn.Module.__getattr__` raises `AttributeError` for unknown names, which `getattr` turns into `None`. `callable` narrows the type, and `_ =` discards the result.

**(f) Installation.** `setattr(cls, "__init__", custom_init)` replaces the constructor on the class. Direct assignment (`cls.__init__ = custom_init`) would work equally well here and also passed basedpyright; `setattr` with a string just makes the runtime patch visible. The decorator returns the same class object, so no wrapper class is created: `type(model) is TinyNet` holds, and pickling round-trips (verified with `pickle.loads(pickle.dumps(...))`).

### 4.8 Summary of what is preserved

| Internal | Mechanism | Where |
|---|---|---|
| Submodule and Parameter registration | `nn.Module.__init__` runs before any field assignment | 4.7 (a) |
| Attribute routing | Fields are assigned with `setattr`, so `nn.Module.__setattr__` dispatches them | 4.7 (d) |
| Hashing (`modules()`, `apply()`, `.to()`) | `__hash__` is not reset | 4.3 (`eq=False`) |
| Identity equality | `__eq__` is not generated | 4.3 (`eq=False`) |
| Layer-tree `repr` | `__repr__` is not generated | 4.3 (`repr=False`) |
| `state_dict` keys | Same modules created in the same order | section 1 check |
| Class identity | Same class object is returned | 4.7 (f) |

---

## 5. Limits and gotchas

- **Do not annotate submodules at class level** unless you want them as constructor arguments (2.3).
- **Mutable class-level defaults.** `dataclass` rejects bare `list` defaults (`mutable default <class 'list'> for field xs is not allowed: use default_factory`). It does not reject a shared `nn.Module` default, because modules are hashable. Verified: with `act: nn.Module = nn.ReLU()`, `S().act is S().act` is `True`, so every instance shares one module. Use `field(default_factory=nn.ReLU)`.
- **Subclasses do not get new constructor fields** (2.4).
- **`kw_only` is supported.** `field(kw_only=True)` fields become keyword-only parameters, as in a dataclass. Positional fields are ordered first.
- **`InitVar` is not supported.** `InitVar` pseudo-fields are not in `fields()`, so the constructor rejects them (`TypeError: too many positional arguments`) and `__post_init__` cannot receive init-only values.
- **Inheritance from other `nn.Module`s with their own `__init__`** is skipped, because the generated `__init__` calls `nn.Module.__init__` directly.

---

## 6. Test coverage

`tests/test_utils.py` covers the behaviour above:

| Test | Checks |
|---|---|
| `test_positional_and_keyword_binding_with_defaults` | Binding and defaults, positional and keyword |
| `test_submodules_are_registered_by_nn_module` | Submodules from `__post_init__` appear in `named_parameters()` (3.1, 4.7 (a)) |
| `test_post_init_runs_after_fields_are_set` | `__post_init__` runs last |
| `test_missing_required_argument_raises` | Custom missing-argument error |
| `test_unexpected_argument_raises` | Native `Signature.bind` error |
| `test_default_factory_is_called_per_instance` | Fresh object per instance (4.6, 4.7 (c)) |
| `test_init_false_field_uses_default_and_is_not_a_parameter` | `init=False` handling (4.7 (c)) |
| `test_rejects_non_module_class` | Runtime guard (4.2) |
| `test_rejects_class_with_own_init` | Runtime guard (4.2) |
| `test_docstring_lists_real_parameters` | Docstring rebuild (4.5) |
| `test_kw_only_field_must_be_passed_by_keyword` | `kw_only` honoured (4.6, section 5) |

Static checking is `basedpyright` in strict mode (`typeCheckingMode = "strict"` in `pyproject.toml`). The decorator and tests are clean under it.

---

## 7. File map

| Path | Role |
|---|---|
| `utils/nn_dataclass.py` | The decorator |
| `utils/__init__.py` | Re-exports `nn_dataclass`, so `from utils import nn_dataclass` works |
| `tests/test_utils.py` | Runtime tests |
| `pyproject.toml` | `[tool.basedpyright]`, strict mode |
