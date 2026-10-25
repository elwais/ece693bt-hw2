from collections.abc import Generator
from torch import nn, inference_mode
from contextlib import contextmanager


@contextmanager
def eval_mode(model: nn.Module) -> Generator[None, None, None]:
    """eval() + inference_mode() as a decorator"""
    was_training = model.training
    model.eval()
    try:
        with inference_mode():
            yield
    finally:
        if was_training:
            model.train()
