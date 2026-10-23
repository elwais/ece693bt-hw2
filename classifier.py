from typing import cast

from torch import nn, Tensor, inference_mode
from utils import built, nn_dataclass


@nn_dataclass
class Classifier(nn.Module):
    # Boilerplate to ensure that forward pass calls are typed appropriately
    # May need to uncomment, but keep hidden for now to make the lesson stick
    def __call__(self, x: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(x))

    # `features` computes the ϕ(x) we'll need for FID
    features: nn.Sequential = built(
        lambda: nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(9216, 128),
            nn.ReLU(),
        )
    )

    # This takes care of the rest
    head: nn.Sequential = built(
        lambda: nn.Sequential(nn.Dropout(0.5), nn.Linear(128, 10))
    )

    @inference_mode()  # Drops autograd computations
    def phi(self, x: Tensor) -> Tensor:
        was_training = self.features.training
        self.features.eval()  # Ensures that dropout is identity, yielding determinism
        try:
            return cast(Tensor, self.features(x))
        finally:
            self.features.train(was_training)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.head(self.features(x)))
