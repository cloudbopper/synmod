"""Model generation"""

import numpy as np
from sklearn.base import RegressorMixin
from mihifepe.simulation.simulation import get_relevant_features

from synmod.models.operations import Average


# pylint: disable = invalid-name
# TODO: add classifier
class Regressor(RegressorMixin):
    """Regression model"""
    operations = [Average]  # TODO: add other ops

    def __init__(self, operation, *args, **kwargs):
        self._operation = operation
        super().__init__(*args, **kwargs)

    def fit(self, X, y):
        """The model is defined in advance, not fitted to data, so throw exception"""
        # TODO: check what happens if this isn't implemented
        raise NotImplementedError("The model is defined in advance, not fitted to data")

    def predict(self, X):
        """Compute outputs on sequences in X by applying property"""
        return self._operation.operate(X)


def get_model(args, features):
    """Generate and return model"""
    # Get relevant features
    relevant_feature_ids = get_relevant_features(args)
    # Select time window for each feature
    windows = [get_window(args.sequence_length) if fid in relevant_feature_ids else None for fid, _ in enumerate(features)]
    # Select model type
    model_type = np.random.choice([Regressor])  # TODO: include classifier
    # Select model operation
    operation = np.random.choice(model_type.operations)(windows)
    # Instantiate and return model
    return model_type(operation)


def get_window(sequence_length):
    """Randomly select appropriate window for model to operate in"""
    # TODO: allow soft-edged windows (smooth decay of influence of feature values outside window)
    right = sequence_length - 1  # Anchor half the windows on the right
    if np.random.random_sample() < 0.5:
        right = np.random.choice(range(sequence_length // 2, sequence_length))
    left = np.random.choice(range(0, right))
    return (left, right)
