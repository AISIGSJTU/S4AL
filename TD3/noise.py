import numpy as np
from typing import Tuple


class GaussianNoise:
    """
    Gaussian Noise Generator.
    Taken from https://github.com/vitchyr/rlkit
    """

    def __init__(
        self,
        action_dim: int,
        min_sigma: float = 1.0,
        max_sigma: float = 1.0,
        decay_period: int = 1000000,
    ):
        self.action_dim = action_dim
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.decay_period = decay_period

    def sample(self, t: int = 0) -> Tuple[float, float]:
        """ Get an action with gaussian noise. """
        sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(
            1.0, t / self.decay_period
        )
        return np.random.normal(0, sigma, size=self.action_dim), sigma
