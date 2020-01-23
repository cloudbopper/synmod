"""Feature generation"""

from abc import ABC

from synmod.constants import BINARY, CATEGORICAL, ORDINAL, CONTINUOUS, STATIC
from synmod.generators import BernoulliProcess, MarkovChain


class StaticFeature(ABC):
    """Feature base class"""
    def __init__(self, name):
        self.name = name

    def sample(self):
        """Sample value for feature"""


class StaticBinaryFeature(StaticFeature):
    """Binary static feature"""
    def __init__(self, name, rng):
        super().__init__(name)
        self._rng = rng
        self.prob = self._rng.uniform()

    def sample(self):
        """Sample value for binary feature"""
        return self._rng.binomial(1, self.prob)


class TemporalFeature(ABC):
    """Temporal feature base class"""
    def __init__(self, name, generator):
        self.name = name
        self._generator = generator

    def sample(self, *args, **kwargs):
        """Sample sequence from generator"""
        return self._generator.sample(*args, **kwargs)


class BinaryFeature(TemporalFeature):
    """Binary feature"""
    def __init__(self, name, rng):
        generator_class = rng.choice([BernoulliProcess, MarkovChain])
        super().__init__(name, generator_class(rng, BINARY))


class CategoricalFeature(TemporalFeature):
    """Categorical feature"""
    def __init__(self, name, rng):
        generator_class = rng.choice([MarkovChain])
        super().__init__(name, generator_class(rng, CATEGORICAL))


class OrdinalFeature(TemporalFeature):
    """Ordinal feature"""
    def __init__(self, name, rng):
        generator_class = rng.choice([MarkovChain])
        super().__init__(name, generator_class(rng, ORDINAL))


class ContinuousFeature(TemporalFeature):
    """Continuous feature"""
    def __init__(self, name, rng):
        generator_class = rng.choice([MarkovChain])
        super().__init__(name, generator_class(rng, CONTINUOUS))


def get_feature(args, name):
    """Return randomly selected feature"""
    if args.synthesis_type == STATIC:
        return StaticBinaryFeature(name, args.rng)
    feature_class = args.rng.choice([BinaryFeature, CategoricalFeature, OrdinalFeature, ContinuousFeature],
                                    p=[1/6, 1/6, 1/6, 1/2])  # noqa: E226
    return feature_class(name, args.rng)
