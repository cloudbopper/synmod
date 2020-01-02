"""Model operations"""

from abc import ABC

import numpy as np


class Operation(ABC):
    """Operation base class"""
    def __init__(self, windows, fop, aop):
        self._windows = windows  # Window corresponding to feature operations
        self._fop = fop  # Feature operator
        self._aop = aop  # Aggregation operator

    # pylint: disable = invalid-name
    def operate(self, X):
        """Operate on input data"""
        # TODO: possibly vectorize use numpy broadcasting for efficiency
        num_sequences, _, fv_length = X.shape
        y = np.empty(num_sequences)
        for sid, sequence in enumerate(X):
            x = np.zeros(fv_length)  # feature-wise outputs
            data = np.transpose(sequence)  # To get features x time
            for ssid, subseq in enumerate(data):
                # Subsequence corresponding to single feature
                window = self._windows[ssid]
                if window is not None:  # Relevant feature
                    (left, right) = window
                    x[ssid] = self._fop(subseq[left: right + 1])  # Apply feature operation in feature-specific window
            y[sid] = self._aop(x)  # Aggregate across features
        return y


class Average(Operation):
    """Computes average of inputs"""
    def __init__(self, windows, polynomial_fn):
        super().__init__(windows, np.average, polynomial_fn)


class Max(Operation):
    """Computes max of inputs"""
    def __init__(self, windows, polynomial_fn):
        super().__init__(windows, np.max, polynomial_fn)
