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
        y = np.empty((len(X), 1))
        for sid, sequence in enumerate(X):
            outputs = {}
            data = np.transpose(sequence)  # To get features x time
            for ssid, subseq in enumerate(data):
                # Subsequence corresponding to single feature
                window = self._windows[ssid]
                if window is not None:  # Relevant feature
                    (left, right) = window
                    outputs[ssid] = self._fop(subseq[left: right + 1])  # Apply feature operation in feature-specific window
            y[sid] = self._aop(list(outputs.values()))  # Aggregate across features
        return y


class Average(Operation):
    """Computes average of inputs"""
    def __init__(self, windows):
        super().__init__(windows, np.average, np.average)
