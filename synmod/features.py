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
    def __init__(self, name, rng, sequence_length):
        self.name = name
        self.window = self.get_window(rng, sequence_length)
        self._generator = None

    def sample(self, *args, **kwargs):
        """Sample sequence from generator"""
        return self._generator.sample(*args, **kwargs)

    @staticmethod
    def get_window(rng, sequence_length):
        """Randomly select a window for the feature where the model should operate in"""
        assert sequence_length is not None  # TODO: handle variable-length sequence case
        # TODO: allow soft-edged windows (smooth decay of influence of feature values outside window)
        right = sequence_length - 1  # Anchor half the windows on the right
        if rng.uniform() < 0.5:
            right = rng.choice(range(sequence_length // 2, sequence_length))
        left = rng.choice(range(0, right))
        return (left, right)


class BinaryFeature(TemporalFeature):
    """Binary feature"""
    def __init__(self, name, rng, sequence_length):
        super().__init__(name, rng, sequence_length)
        generator_class = rng.choice([BernoulliProcess, MarkovChain])
        self._generator = generator_class(rng, BINARY, self.window)


class CategoricalFeature(TemporalFeature):
    """Categorical feature"""
    def __init__(self, name, rng, sequence_length):
        super().__init__(name, rng, sequence_length)
        generator_class = rng.choice([MarkovChain])
        self._generator = generator_class(rng, CATEGORICAL, self.window)


class OrdinalFeature(TemporalFeature):
    """Ordinal feature"""
    def __init__(self, name, rng, sequence_length):
        super().__init__(name, rng, sequence_length)
        generator_class = rng.choice([MarkovChain])
        self._generator = generator_class(rng, ORDINAL, self.window)


class ContinuousFeature(TemporalFeature):
    """Continuous feature"""
    def __init__(self, name, rng, sequence_length):
        super().__init__(name, rng, sequence_length)
        generator_class = rng.choice([MarkovChain])
        self._generator = generator_class(rng, CONTINUOUS, self.window)


def get_feature(args, name):
    """Return randomly selected feature"""
    if args.synthesis_type == STATIC:
        return StaticBinaryFeature(name, args.rng)
    feature_class = args.rng.choice([BinaryFeature, CategoricalFeature, OrdinalFeature, ContinuousFeature],
                                    p=[1/6, 1/6, 1/6, 1/2])  # noqa: E226
    return feature_class(name, args.rng, args.sequence_length)
