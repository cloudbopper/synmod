"""Feature generation"""

from abc import ABC
from copy import deepcopy

from synmod.constants import BINARY, CATEGORICAL, CONTINUOUS, STATIC
from synmod.generators import BernoulliProcess, MarkovChain
from synmod.aggregators import Max, get_aggregation_fn


class Feature(ABC):
    """Feature base class"""
    def __init__(self, name):
        self.name = name

    def sample(self, *args, **kwargs):
        """Sample value for feature"""

    def summary(self):
        """Return dictionary summarizing feature"""
        return dict(name=self.name,
                    type=self.__class__.__name__)


class StaticBinaryFeature(Feature):
    """Binary static feature"""
    def __init__(self, name, rng):
        super().__init__(name)
        self._rng = rng
        self.prob = self._rng.uniform()

    def sample(self, *args, **kwargs):
        """Sample value for binary feature"""
        return self._rng.binomial(1, self.prob)

    def summary(self):
        summary = super().summary()
        summary.update(dict(prob=self.prob))
        return summary


class TemporalFeature(Feature):
    """Base class for features that take a sequence of values"""
    def __init__(self, name, rng, sequence_length, aggregation_fn):
        super().__init__(name)
        self.window = self.get_window(rng, sequence_length)
        self.generator = None
        self.aggregation_fn = aggregation_fn

    def sample(self, *args, **kwargs):
        """Sample sequence from generator"""
        return self.generator.sample(*args, **kwargs)

    def summary(self):
        summary = super().summary()
        assert self.generator is not None
        summary.update(dict(window=self.window,
                            aggregation_fn=self.aggregation_fn.__class__.__name__,
                            generator=self.generator.summary()))
        return summary

    @staticmethod
    def get_window(rng, sequence_length):
        """Randomly select a window for the feature where the model should operate in"""
        assert sequence_length is not None  # TODO: handle variable-length sequence case
        # TODO: allow soft-edged windows (smooth decay of influence of feature values outside window)
        right = rng.choice(range(sequence_length // 2, sequence_length))
        left = rng.choice(range(0, right))
        return (left, right)


class BinaryFeature(TemporalFeature):
    """Binary feature"""
    def __init__(self, name, rng, sequence_length, aggregation_fn, **kwargs):
        super().__init__(name, rng, sequence_length, aggregation_fn)
        generator_class = rng.choice([BernoulliProcess, MarkovChain])
        kwargs["n_states"] = 2
        self.generator = generator_class(rng, BINARY, self.window, **kwargs)


class CategoricalFeature(TemporalFeature):
    """Categorical feature"""
    def __init__(self, name, rng, sequence_length, aggregation_fn, **kwargs):
        super().__init__(name, rng, sequence_length, aggregation_fn)
        generator_class = rng.choice([MarkovChain])
        kwargs["n_states"] = kwargs.get("n_states", rng.integers(3, 5, endpoint=True))
        self.generator = generator_class(rng, CATEGORICAL, self.window, **kwargs)


class ContinuousFeature(TemporalFeature):
    """Continuous feature"""
    def __init__(self, name, rng, sequence_length, aggregation_fn, **kwargs):
        super().__init__(name, rng, sequence_length, aggregation_fn)
        generator_class = rng.choice([MarkovChain])
        self.generator = generator_class(rng, CONTINUOUS, self.window, **kwargs)


def get_feature(args, name):
    """Return randomly selected feature"""
    if args.synthesis_type == STATIC:
        return StaticBinaryFeature(name, args.rng)
    aggregation_fn = get_aggregation_fn(args.rng)
    kwargs = {"window_independent": args.window_independent}
    feature_class = args.rng.choice([BinaryFeature, CategoricalFeature, ContinuousFeature], p=[1/4, 1/4, 1/2])  # noqa: E226
    if isinstance(aggregation_fn, Max):
        # Avoid low-variance features by sampling continuous or high-state-count categorical feature
        feature_class = args.rng.choice([CategoricalFeature, ContinuousFeature], p=[1/4, 3/4])  # noqa: E226
        if feature_class == CategoricalFeature:
            kwargs["n_states"] = args.rng.integers(4, 5, endpoint=True)
    feature = feature_class(name, deepcopy(args.rng), args.sequence_length, aggregation_fn, **kwargs)
    args.logger.info(f"Generating feature class {feature_class.__name__} with window {feature.window} and"
                     f" aggregation_fn {aggregation_fn.__class__.__name__}")
    return feature
