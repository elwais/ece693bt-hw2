from collections.abc import Generator
from torch import nn, no_grad
from contextlib import contextmanager


@contextmanager
def eval_mode(model: nn.Module) -> Generator[None, None, None]:
    """eval() + no_grad() as a decorator"""
    was_training = model.training
    model.eval()
    try:
        with no_grad():
            yield
    finally:
        if was_training:
            model.train()
