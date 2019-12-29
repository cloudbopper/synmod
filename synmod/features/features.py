"""Feature generation"""

import numpy as np
from mihifepe.feature import Feature

from synmod.constants import DISCRETE, CONTINUOUS
from synmod.features.generators import BernoulliProcess

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
        # TODO: add other choices of generating processes
        generator_class = np.random.choice([BernoulliProcess])
        self._generator = generator_class(self.rng_seed)


def get_feature(name):
    """Return randomly selected feature"""
    # TODO: generate mix of discrete and continuous features
    # feature_class = np.random.choice(FEATURE_TYPES)
    # if feature_class == CONTINUOUS:
    #     return ContinousFeature
    # else:
    #     return np.random.choice(BinaryFeature, CategoricalFeature, OrdinalFeature)
    feature_class = BinaryFeature
    return feature_class(name)
