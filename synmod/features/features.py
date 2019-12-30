"""Feature generation"""

from mihifepe.feature import Feature

from synmod.constants import BINARY, CATEGORICAL, ORDINAL, CONTINUOUS
from synmod.features.generators import BernoulliProcess, MarkovChain


# TODO: shouldn't be able to instantiate this directly
class TemporalFeature(Feature):
    """Temporal feature"""
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self._generator = None

    def sample(self, *args, **kwargs):
        """Sample sequence from generator"""
        return self._generator.sample(*args, **kwargs)


class BinaryFeature(TemporalFeature):
    """Binary feature"""
    def __init__(self, name, rng, **kwargs):
        super().__init__(name, **kwargs)
        generator_class = rng.choice([BernoulliProcess, MarkovChain])
        self._generator = generator_class(rng, BINARY)


class CategoricalFeature(TemporalFeature):
    """Categorical feature"""
    def __init__(self, name, rng, **kwargs):
        super().__init__(name, **kwargs)
        generator_class = rng.choice([MarkovChain])
        self._generator = generator_class(rng, CATEGORICAL)


class OrdinalFeature(TemporalFeature):
    """Ordinal feature"""
    def __init__(self, name, rng, **kwargs):
        super().__init__(name, **kwargs)
        generator_class = rng.choice([MarkovChain])
        self._generator = generator_class(rng, ORDINAL)


class ContinuousFeature(TemporalFeature):
    """Continuous feature"""
    def __init__(self, name, rng, **kwargs):
        super().__init__(name, **kwargs)
        generator_class = rng.choice([MarkovChain])
        self._generator = generator_class(rng, CONTINUOUS)


def get_feature(args, name):
    """Return randomly selected feature"""
    # TODO: generate mix of discrete and continuous features
    feature_class = args.rng.choice([BinaryFeature, CategoricalFeature, OrdinalFeature, ContinuousFeature],
                                    p=[1/6, 1/6, 1/6, 1/2])
    return feature_class(name, args.rng)
