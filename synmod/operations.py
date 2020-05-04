"""Model operations"""

from abc import ABC

import numpy as np


class Operation(ABC):
    """Operation base class"""
    def __init__(self, operator=None):
        self._operator = operator  # Feature temporal aggregation function
        self._windows = None

    def operate_on_feature(self, sequences):
        """Operate on sequences for given feature"""
        return np.apply_along_axis(self._operator, 1, sequences)  # sequences: instances X timesteps

    def set_windows(self, windows):
        """Set windows to operate on for all features"""
        self._windows = windows  # Windows over time for all features (list of tuples)

    def operate(self, sequences):
        """Apply feature-wise operations to sequence data"""
        # TODO: when perturbing a feature, other values do not need to be recomputed.
        # But this seems unavoidable under the current design (analysis only calls model.predict, doesn't provide other info)
        num_instances, num_features, _ = sequences.shape  # sequences: instances X features X timesteps
        matrix = np.zeros((num_instances, num_features))
        for fidx in range(num_features):
            window = self._windows[fidx]
            if window is not None:
                # Relevant feature; if irrelevant, values can be zero as they don't matter
                (left, right) = window
                matrix[:, fidx] = self.operate_on_feature(sequences[:, fidx, left: right + 1])
        return matrix


class Average(Operation):
    """Computes average of inputs"""
    def __init__(self):
        super().__init__(np.average)


class Max(Operation):
    """Computes max of inputs"""
    def __init__(self):
        super().__init__(np.max)


class Identity(Operation):
    """Returns the input as-is, does not aggregate (used for static data)"""
    def operate(self, sequences):
        return sequences
