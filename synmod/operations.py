"""Model operations"""

from abc import ABC

import numpy as np


class Operation(ABC):
    """Operation base class"""
    def __init__(self, windows=None, operator=None):
        self._windows = windows  # Window corresponding to feature operations
        self._operator = operator  # Feature operator

    def operate(self, sequences):
        """Apply feature-wise operations to sequence data"""
        # TODO: possibly vectorize using masks for efficiency
        # TODO: when perturbing a feature, other values do not need to be recomputed
        num_instances, num_features, _ = sequences.shape  # sequences: instances X features X timesteps
        matrix = np.zeros((num_instances, num_features))
        for fidx in range(num_features):
            window = self._windows[fidx]
            if window is not None:
                (left, right) = window
                matrix[:, fidx] = np.apply_along_axis(self._operator, 1, sequences[:, fidx, left: right + 1])
        return matrix


class Average(Operation):
    """Computes average of inputs"""
    def __init__(self, windows):
        super().__init__(windows, np.average)


class Max(Operation):
    """Computes max of inputs"""
    def __init__(self, windows):
        super().__init__(windows, np.max)


class Identity(Operation):
    """Returns the input as-is, does not aggregate (used for static data)"""
    def operate(self, sequences):
        return sequences
