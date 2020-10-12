"""Feature temporal aggregation functions"""

from abc import ABC

import numpy as np


class Aggregator():
    """Aggregates temporal values"""
    def __init__(self, aggregation_fns, windows, instances=None, standardize_features=False):
        self._aggregation_fns = aggregation_fns  # Temporal aggregation functions for all features
        self._windows = windows  # Windows over time for all features (list of tuples)
        if instances is None:
            return
        # Identify statistics to standardize each feature
        num_features = len(self._aggregation_fns)
        self._means = np.zeros(num_features)
        self._stds = np.ones(num_features)
        if standardize_features:
            self.update_statistics(instances)

    def update_statistics(self, instances):
        """Identify statistics to standardize each feature"""
        for fidx, _ in enumerate(self._aggregation_fns):
            left, right = self._windows[fidx]
            vec = self.operate_on_feature(fidx, instances[:, fidx, left: right + 1])
            self._means[fidx] = np.mean(vec)
            self._stds[fidx] = np.std(vec)
            if self._stds[fidx] < 1e-10:
                # FIXME: features can pass the variance test earlier but fail it here, since the samples used are different
                self._stds[fidx] = 1

    def operate_on_feature(self, fidx, sequences):
        """Operate on sequences for given feature"""
        return (self._aggregation_fns[fidx].operate(sequences) - self._means[fidx]) / self._stds[fidx]  # sequences: instances X timesteps

    def operate(self, sequences):
        """Apply feature-wise operations to sequence data"""
        # TODO: when perturbing a feature, other values do not need to be recomputed.
        # But this seems unavoidable under the current design (analysis only calls model.predict, doesn't provide other info)
        num_instances, num_features, _ = sequences.shape  # sequences: instances X features X timesteps
        matrix = np.zeros((num_instances, num_features))
        for fidx in range(num_features):
            (left, right) = self._windows[fidx]
            matrix[:, fidx] = self.operate_on_feature(fidx, sequences[:, fidx, left: right + 1])
        return matrix


class StaticAggregator(Aggregator):
    """Returns input as-is without aggregation (for static features)"""
    def __init__(self):
        super().__init__(None, None)

    def operate(self, sequences):
        return sequences


class AggregationFunction(ABC):
    """Aggregation function base class"""
    def __init__(self, operator):
        self._operator = operator

    def operate(self, sequences):
        """Operate on sequences for given feature"""
        return np.apply_along_axis(self._operator, 1, sequences)  # sequences: instances X timesteps


class Slope(AggregationFunction):
    """Computes slope of inputs"""
    def __init__(self):
        super().__init__(lambda seq: (seq[-1] - seq[0]) / seq.shape[0])


class Average(AggregationFunction):
    """Computes average of inputs"""
    def __init__(self):
        super().__init__(np.average)


class Max(AggregationFunction):
    """Computes max of inputs"""
    def __init__(self):
        super().__init__(np.max)


def get_aggregation_fn(rng):
    """Sample aggregation function for feature"""
    return rng.choice([Average, Max, Slope])()
