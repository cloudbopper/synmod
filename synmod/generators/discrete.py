"""Generators for discrete features"""

from scipy.stats import bernoulli

from synmod.generators.generator import Generator

# pylint: disable = too-few-public-methods
class BernoulliProcess(Generator):
    """Bernoulli process generator"""
    def __init__(self, seed, **kwargs):
        super().__init__(seed)
        self._p = kwargs.get("p", self._rng.uniform(0.01, 0.99))

    def sample(self, sequence_length):
        return bernoulli.rvs(p=self._p, size=sequence_length, random_state=self._rng)