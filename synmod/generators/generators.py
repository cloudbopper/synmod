"""Feature generators"""

from mihifepe.feature import Feature
import numpy as np

from synmod.constants import DISCRETE, CONTINUOUS
from synmod.generators.generator import Generator
from synmod.generators.discrete import *
from synmod.generators.continuous import *

FEATURE_TYPES = [DISCRETE, CONTINUOUS]

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
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        generator_class = np.random.choice([BernoulliProcess])
        self._generator = generator_class(self.rng_seed)


def get_feature(name):
    """Return randomly selected feature"""
    feature_class = get_feature_class()
    return feature_class(name)


def get_feature_class():
    """Get feature class"""
    # TODO: allow non-binary feature classes
    # feature_class = np.random.choice(FEATURE_TYPES)
    # if feature_class == CONTINUOUS:
    #     return ContinousFeature
    # else:
    #     return np.random.choice(BinaryFeature, CategoricalFeature, OrdinalFeature)
    return BinaryFeature