"""Model operations"""

from abc import ABC

import numpy as np


class Operation(ABC):
    """Operation base class"""
    def __init__(self, windows, operator):
        self._windows = windows  # Window corresponding to feature operations
        self._operator = operator  # Feature operator

    def operate(self, sequences):
        """Apply feature-wise operations to sequence data"""
        # TODO: possibly vectorize use numpy broadcasting for efficiency
        num_sequences, _, num_features = sequences.shape  # num_sequences * seq_length * num_features
        matrix = np.empty((num_sequences, num_features))
        for sid, sequence in enumerate(sequences):
            row = np.zeros(num_features)  # feature-wise outputs
            data = np.transpose(sequence)  # To get features x time
            for ssid, subseq in enumerate(data):
                # Subsequence corresponding to single feature
                window = self._windows[ssid]
                if window is not None:  # Relevant feature
                    (left, right) = window
                    row[ssid] = self._operator(subseq[left: right + 1])  # Apply feature-specific operation in feature-specific window
            matrix[sid] = row
        return matrix


class Average(Operation):
    """Computes average of inputs"""
    def __init__(self, windows):
        super().__init__(windows, np.average)


class Max(Operation):
    """Computes max of inputs"""
    def __init__(self, windows):
        super().__init__(windows, np.max)
