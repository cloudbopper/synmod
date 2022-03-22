"""Feature generation"""

from abc import ABC

import numpy as np

from synmod.constants import BINARY, CATEGORICAL, NUMERIC, TABULAR
from synmod.generators import BernoulliDistribution, CategoricalDistribution, NormalDistribution
from synmod.generators import BernoulliProcess, MarkovChain
from synmod.aggregators import Max, get_aggregation_fn_cls


class Feature(ABC):
    """Feature base class"""
    def __init__(self, name, seed_seq):
        self.name = name
        self._rng = np.random.default_rng(seed_seq)
        # Initialize relevance
        self.important = False
        self.effect_size = 0

    def sample(self, *args, **kwargs):
        """Sample value for feature"""

    def summary(self):
        """Return dictionary summarizing feature"""
        return dict(name=self.name,
                    type=self.__class__.__name__)


class TabularFeature(Feature):
    """Tabular feature"""
    def __init__(self, name, seed_seq):
        super().__init__(name, seed_seq)
        self.generator = None

    def sample(self, *args, **kwargs):
        """Sample value from generator"""
        return self.generator.sample()

    def summary(self):
        summary = super().summary()
        summary.update(dict(generator=self.generator.summary()))
        return summary


class TabularBinaryFeature(TabularFeature):
    """Tabular binary feature"""
    def __init__(self, name, seed_seq):
        super().__init__(name, seed_seq)
        self.generator = BernoulliDistribution(self._rng)


class TabularCategoricalFeature(TabularFeature):
    """Tabular binary feature"""
    def __init__(self, name, seed_seq):
        super().__init__(name, seed_seq)
        self.generator = CategoricalDistribution(self._rng)


class TabularNumericFeature(TabularFeature):
    """Tabular binary feature"""
    def __init__(self, name, seed_seq):
        super().__init__(name, seed_seq)
        generator_class = self._rng.choice([NormalDistribution])
        self.generator = generator_class(self._rng)


class TemporalFeature(Feature):
    """Base class for features that take a sequence of values"""
    def __init__(self, name, seed_seq, sequence_length, aggregation_fn_cls):
        super().__init__(name, seed_seq)
        self.window = self.get_window(sequence_length)
        self.generator = None
        self.aggregation_fn = aggregation_fn_cls(**dict(rng=self._rng, window=self.window))
        # Initialize relevance
        self.window_important = False
        self.ordering_important = False
        self.window_ordering_important = False

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

    def get_window(self, sequence_length):
        """Randomly select a window for the feature where the model should operate in"""
        assert sequence_length is not None  # TODO: handle variable-length sequence case
        if sequence_length == 1:
            return (1, 1)  # tabular features
        # TODO: allow soft-edged windows (smooth decay of influence of feature values outside window)
        right = self._rng.choice(range(sequence_length // 2, sequence_length))
        left = self._rng.choice(range(0, right))
        return (left, right)


class BinaryFeature(TemporalFeature):
    """Binary feature"""
    def __init__(self, name, seed_seq, sequence_length, aggregation_fn_cls, **kwargs):
        super().__init__(name, seed_seq, sequence_length, aggregation_fn_cls)
        generator_class = self._rng.choice([BernoulliProcess, MarkovChain])
        kwargs["n_states"] = 2
        self.generator = generator_class(self._rng, BINARY, self.window, **kwargs)


class CategoricalFeature(TemporalFeature):
    """Categorical feature"""
    def __init__(self, name, seed_seq, sequence_length, aggregation_fn_cls, **kwargs):
        super().__init__(name, seed_seq, sequence_length, aggregation_fn_cls)
        generator_class = self._rng.choice([MarkovChain])
        kwargs["n_states"] = kwargs.get("n_states", self._rng.integers(3, 5, endpoint=True))
        self.generator = generator_class(self._rng, CATEGORICAL, self.window, **kwargs)


class NumericFeature(TemporalFeature):
    """Numeric feature"""
    def __init__(self, name, seed_seq, sequence_length, aggregation_fn_cls, **kwargs):
        super().__init__(name, seed_seq, sequence_length, aggregation_fn_cls)
        generator_class = self._rng.choice([MarkovChain])
        self.generator = generator_class(self._rng, NUMERIC, self.window, **kwargs)


def get_feature(args, name):
    """Return randomly selected feature"""
    seed_seq = args.rng.bit_generator._seed_seq.spawn(1)[0]  # pylint: disable = protected-access
    if args.synthesis_type == TABULAR:
        feature_class = args.rng.choice([TabularBinaryFeature, TabularCategoricalFeature, TabularNumericFeature],
                                        p=args.feature_type_distribution)
        args.logger.info(f"Generating feature class {feature_class.__name__}")
        return feature_class(name, seed_seq)
    else:
        aggregation_fn_cls = get_aggregation_fn_cls(args.rng)
        kwargs = {"window_independent": args.window_independent}
        feature_class = args.rng.choice([BinaryFeature, CategoricalFeature, NumericFeature], p=args.feature_type_distribution)
        if aggregation_fn_cls is Max:
            # Avoid low-variance features by sampling numeric or high-state-count categorical feature
            feature_class = args.rng.choice([CategoricalFeature, NumericFeature], p=[0.25, 0.75])
            if feature_class == CategoricalFeature:
                kwargs["n_states"] = args.rng.integers(4, 5, endpoint=True)
        feature = feature_class(name, seed_seq, args.sequence_length, aggregation_fn_cls, **kwargs)
        args.logger.info(f"Generating feature class {feature_class.__name__} with window {feature.window} and"
                         f" aggregation_fn {aggregation_fn_cls.__name__}")
        return feature
