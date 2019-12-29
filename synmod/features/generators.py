"""Generator base class"""

from abc import ABC

import numpy as np
from scipy.stats import bernoulli


class Generator(ABC):
    """Generator base class"""
    def __init__(self, seed=np.random.randint):
        self._rng = np.random.RandomState(seed)

    def sample(self, sequence_length):
        """Sample sequence of given length from generator"""


class BernoulliProcess(Generator):
    """Bernoulli process generator"""
    def __init__(self, seed, **kwargs):
        super().__init__(seed)
        self._p = kwargs.get("p", self._rng.uniform(0.01, 0.99))

    def sample(self, sequence_length):
        return bernoulli.rvs(p=self._p, size=sequence_length, random_state=self._rng)
