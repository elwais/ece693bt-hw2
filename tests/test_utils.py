from dataclasses import field

import pytest
import torch.nn as nn

from utils import built, nn_dataclass


@nn_dataclass
class Block(nn.Module):
    width: int
    depth: int = 2
    tags: list[str] | None = None

    def __post_init__(self) -> None:
        self.proj: nn.Linear = nn.Linear(self.width, self.width)
        self.seen_post_init: bool = True


def test_positional_and_keyword_binding_with_defaults() -> None:
    a = Block(4)
    assert (a.width, a.depth, a.tags) == (4, 2, None)
    b = Block(4, depth=3, tags=["x"])
    assert (b.depth, b.tags) == (3, ["x"])


def test_submodules_are_registered_by_nn_module() -> None:
    blk = Block(4)
    names = {n for n, _ in blk.named_parameters()}
    assert names == {"proj.weight", "proj.bias"}


def test_post_init_runs_after_fields_are_set() -> None:
    assert Block(3).seen_post_init


def test_missing_required_argument_raises() -> None:
    with pytest.raises(TypeError, match="missing required argument: 'width'"):
        _ = Block()  # pyright: ignore[reportCallIssue]


def test_unexpected_argument_raises() -> None:
    with pytest.raises(TypeError):
        _ = Block(4, nope=1)  # pyright: ignore[reportCallIssue]


def test_default_factory_is_called_per_instance() -> None:
    @nn_dataclass
    class WithFactory(nn.Module):
        items: list[int] = field(default_factory=list[int])

    a, b = WithFactory(), WithFactory()
    a.items.append(1)
    assert b.items == []


def test_init_false_field_uses_default_and_is_not_a_parameter() -> None:
    @nn_dataclass
    class Counter(nn.Module):
        start: int = 0
        count: int = field(default=10, init=False)

    c = Counter(5)
    assert (c.start, c.count) == (5, 10)
    with pytest.raises(TypeError):
        _ = Counter(5, 7)  # pyright: ignore[reportCallIssue]



def test_built_field_is_fresh_per_instance_and_not_a_parameter() -> None:
    @nn_dataclass
    class Net(nn.Module):
        width: int = 4
        head: nn.Linear = built(lambda: nn.Linear(2, 2))

    a, b = Net(), Net(8)
    assert a.head is not b.head  # each instance builds its own module
    assert "head.weight" in dict(a.named_parameters())  # registered as a submodule
    with pytest.raises(TypeError):
        _ = Net(head=nn.Linear(2, 2))  # pyright: ignore[reportCallIssue]



def test_built_rejects_a_module_instead_of_a_factory() -> None:
    with pytest.raises(TypeError, match="factory, not a module"):
        # A module type-checks as Callable[..., Any], so T is inferred as Any.
        _ = built(nn.Linear(2, 2))  # pyright: ignore[reportAny]


class NotAModule:
    pass


def test_rejects_non_module_class() -> None:
    with pytest.raises(TypeError, match="nn.Module subclasses"):
        nn_dataclass(NotAModule)  # pyright: ignore[reportArgumentType]


def test_rejects_class_with_own_init() -> None:
    class Custom(nn.Module):
        def __init__(self) -> None:
            super().__init__()

    with pytest.raises(TypeError, match="defines __init__"):
        _ = nn_dataclass(Custom)


def test_docstring_lists_real_parameters() -> None:
    assert Block.__doc__ == "Block(width: int, depth: int = 2, tags: list[str] | None = None)"


def test_kw_only_field_must_be_passed_by_keyword() -> None:
    @nn_dataclass
    class KwOnly(nn.Module):
        a: int = 1
        b: int = field(default=2, kw_only=True)

    assert KwOnly(5, b=7).b == 7
    with pytest.raises(TypeError):
        _ = KwOnly(1, 5)  # pyright: ignore[reportCallIssue]
