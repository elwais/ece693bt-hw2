from collections.abc import Generator
from contextlib import contextmanager

from torch import inference_mode, nn


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
